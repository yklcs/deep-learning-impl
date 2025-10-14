from typing import Tuple

import torch
from torch import Tensor


# Step 1: Define custom operators using torch.library API
@torch.library.custom_op(
    "my_ops::batchnorm_forward", mutates_args=("running_mean", "running_var")
)
def batchnorm_forward(
    input: Tensor,  # [N, C, H, W]
    gamma: Tensor,  # [C]
    beta: Tensor,  # [C]
    running_mean: Tensor,  # [C]
    running_var: Tensor,  # [C]
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """forward pass of BatchNorm for 4D input [N, C, H, W]."""

    n, _c, h, w = input.shape
    reduce_dims = (0, 2, 3)
    expand_shape = (1, -1, 1, 1)

    # Ref. https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    if training:
        save_mean = input.mean(dim=reduce_dims)
        var = input.var(dim=reduce_dims, unbiased=False)
        m = n * h * w
        var_unbiased = (m / (m - 1)) * var

        running_mean[:] = (1 - momentum) * running_mean + momentum * save_mean
        running_var[:] = (1 - momentum) * running_var + momentum * var_unbiased
    else:
        save_mean = running_mean.clone()
        var = running_var.clone()

    save_invstd = torch.rsqrt(var + eps)
    norm_input = (input - save_mean.view(expand_shape)) * save_invstd.view(expand_shape)
    output = gamma.view(expand_shape) * norm_input + beta.view(expand_shape)

    return output, save_mean, save_invstd


@torch.library.custom_op("my_ops::batchnorm_backward", mutates_args=())
def batchnorm_backward(
    grad_output: Tensor,  # [N, C, H, W]
    input: Tensor,  # [N, C, H, W]
    gamma: Tensor,  # [C]
    save_mean: Tensor,  # [C]
    save_invstd: Tensor,  # [C]
) -> Tuple[Tensor, Tensor, Tensor]:
    """backward pass of BatchNorm for 4D input."""

    # Implement Here
    # Ref. https://arxiv.org/abs/1502.03167

    reduce_dims = (0, 2, 3)
    expand_shape = (1, -1, 1, 1)

    norm_input = (input - save_mean.view(expand_shape)) * save_invstd.view(expand_shape)

    grad_gamma = (grad_output * norm_input).sum(dim=reduce_dims)
    grad_beta = grad_output.sum(dim=reduce_dims)
    grad_input = (
        gamma.view(expand_shape)
        * save_invstd.view(expand_shape)
        * (
            grad_output
            - grad_output.mean(dim=reduce_dims, keepdim=True)
            - norm_input
            * (grad_output * norm_input).mean(dim=reduce_dims, keepdim=True)
        )
    )

    return grad_input, grad_gamma, grad_beta


# Step 2: Connect forward and backward with autograd
# This connects our custom forward/backward operators to PyTorch's
# autograd system, allowing gradients to flow during backpropagation
class BatchNormCustom(torch.autograd.Function):
    """
    Custom Batch Normalization for 4D inputs [N, C, H, W].

    Bridges custom operators with PyTorch's autograd engine.
    - forward(): calls custom forward operator and saves context
    - backward(): calls custom backward operator using saved context

    Usage:
        output = BatchNormCustom.apply(input, gamma, beta, running_mean, running_var, training, momentum, eps)
    """

    @staticmethod
    def forward(
        ctx, input, gamma, beta, running_mean, running_var, training, momentum, eps
    ):
        output, save_mean, save_invstd = torch.ops.my_ops.batchnorm_forward(
            input, gamma, beta, running_mean, running_var, training, momentum, eps
        )
        ctx.save_for_backward(input, gamma, save_mean, save_invstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, save_mean, save_invstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = torch.ops.my_ops.batchnorm_backward(
            grad_output, input, gamma, save_mean, save_invstd
        )
        # Return gradients for all forward inputs (None for non-tensor args)
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None
