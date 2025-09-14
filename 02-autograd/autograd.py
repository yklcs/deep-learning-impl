import numpy as np
import torch
from numpy.testing import assert_allclose


class Context:
    """Context to save variables for backward computation."""

    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tuple(tensors)


class Op:
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs):
        ctx = Context()
        raw_inputs = [x.data if isinstance(x, Tensor) else x for x in inputs]
        result_data = cls.forward(ctx, *raw_inputs)

        requires_grad = any(isinstance(x, Tensor) and x.requires_grad for x in inputs)
        out = Tensor(result_data, requires_grad=requires_grad)
        if requires_grad:
            out.grad_fn = (cls, ctx, inputs)
        return out


def reduce_grad_to_shape(grad, shape):
    """
    Match the gradient dimension for the input tensor
    - grad: output gradient
    - shape: original input tensor shape
    """
    # remove expanded dimension for batched data
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # remove the broadcasted axis
    for i, dim in enumerate(shape):
        if dim == 1 and (grad.shape[i] != 1):
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Add(Op):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a.shape, b.shape)
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        grad_a = reduce_grad_to_shape(grad_output, a_shape)
        grad_b = reduce_grad_to_shape(grad_output, b_shape)
        return grad_a, grad_b


class Mul(Op):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = reduce_grad_to_shape(b * grad_output, a.shape)
        grad_b = reduce_grad_to_shape(a * grad_output, b.shape)
        return grad_a, grad_b


class MatMul(Op):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        return grad_a, grad_b


class Pow(Op):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a**b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output * a ** (b - 1) * b
        grad_b = grad_output * a**b * np.log(a)
        return grad_a, grad_b


class Sum(Op):
    @staticmethod
    def forward(ctx, a, axis=None, keepdims=False):
        ctx.save_for_backward(a.shape, axis, keepdims)
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, grad_output):
        (a_shape, axis, keepdims) = ctx.saved_tensors

        # Restore reduced dimensions
        if not keepdims and axis is not None:
            to_expand = axis
            grad_output = np.expand_dims(grad_output, to_expand)

        grad_a = np.broadcast_to(grad_output, a_shape)
        return grad_a


class ReLU(Op):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.maximum(a, 0)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        grad_a = (a > 0) * grad_output
        return reduce_grad_to_shape(grad_a, a.shape)


class CrossEntropyLoss(Op):
    @staticmethod
    def forward(ctx, logits, targets):
        assert logits.ndim <= 2, "Multidimensional classes not supported"

        # Log-Softmax, see https://arxiv.org/pdf/1909.03469
        logits_max = logits.max(axis=-1, keepdims=True)
        logits_shifted = logits - logits_max
        logsumexp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))
        log_probs = logits_shifted - logsumexp

        losses = -np.sum(targets * log_probs, axis=-1)
        ctx.save_for_backward(log_probs, targets, np.size(losses))

        return np.mean(losses)

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, batch_size = ctx.saved_tensors

        grad_logits = (
            grad_output
            * (targets.sum(-1, keepdims=True) * np.exp(log_probs) - targets)
            / batch_size
        )
        grad_targets = grad_output * -log_probs / batch_size
        return grad_logits, grad_targets


class Log(Op):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_x = grad_output * (1.0 / x)
        return reduce_grad_to_shape(grad_x, x.shape)


class NLLLoss(Op):
    @staticmethod
    def forward(ctx, log_probs, targets_one_hot):
        assert log_probs.ndim <= 2, "Multidimensional classes not supported"

        # PyTorch batches on dim 0 or 1
        losses = -np.sum(targets_one_hot * log_probs, axis=-1)
        ctx.save_for_backward(log_probs, targets_one_hot, np.size(losses))

        return np.mean(losses)

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets_one_hot, batch_size = ctx.saved_tensors

        grad_log_probs = grad_output * -targets_one_hot / batch_size
        grad_targets = grad_output * -log_probs / batch_size

        return grad_log_probs, grad_targets


class Softmax(Op):
    @staticmethod
    def forward(ctx, logits):
        # Shifting for stability, see https://arxiv.org/pdf/1909.03469
        logits_max = logits.max(axis=-1, keepdims=True)
        logits_exp_shifted = np.exp(logits - logits_max)
        probs = logits_exp_shifted / logits_exp_shifted.sum(axis=-1, keepdims=True)
        ctx.save_for_backward(probs)

        return probs

    @staticmethod
    def backward(ctx, grad_output):
        (probs,) = ctx.saved_tensors
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        # https://github.com/jax-ml/jax/blob/f74467851b1186b434d4b538d0be419378a47a69/jax/_src/nn/functions.py#L648-L652
        return probs * (grad_output - (probs * grad_output).sum(axis=-1, keepdims=True))


class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, Tensor):
            data = data.data  # unwrap
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None  # (OpClass, ctx, inputs) for non-leaf tensors

    # Operator overloads delegate to our Op classes via apply:
    def __add__(self, other):
        return Add.apply(self, other)

    def __radd__(self, other):
        return Add.apply(other, self)

    def __sub__(self, other):
        return Add.apply(self, Mul.apply(other, Tensor(-1.0)))

    def __rsub__(self, other):
        return Add.apply(other, Mul.apply(self, Tensor(-1.0)))

    def __mul__(self, other):
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return Mul.apply(other, self)

    def __matmul__(self, other):
        return MatMul.apply(self, other)  # matrix @

    def __neg__(self):
        return Mul.apply(self, Tensor(-1.0))

    def __truediv__(self, other):
        return Mul.apply(self, Tensor(1.0) / other)

    def __pow__(self, exponent):
        return Pow.apply(self, exponent)

    def __rpow__(self, base):
        return Pow.apply(base, self)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

    # initialization
    @staticmethod
    def zeros(shape, requires_grad=False):
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

    @staticmethod
    def ones(shape, requires_grad=False):
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

    def backward(self, grad_output=None):
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call backward on a tensor that does not require grad."
            )
        # If no grad specified, tensor must be scalar
        if grad_output is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad_output must be provided for non-scalar tensors"
                )
            grad_output = np.ones_like(self.data, dtype=np.float32)
        else:
            grad_output = np.array(grad_output, dtype=np.float32)
        # Initialize this tensor's gradient
        self.grad = grad_output

        # Build a topologically sorted list of tensors (post-order DFS)
        topo_order = []
        visited = set()

        def build_graph(t):
            if isinstance(t, Tensor) and t not in visited:
                visited.add(t)
                if t.grad_fn is not None:  # not a leaf
                    op_cls, ctx, inputs = t.grad_fn
                    for inp in inputs:
                        build_graph(inp)
                topo_order.append(t)

        build_graph(self)

        # Traverse graph in reverse topological order, apply chain rule
        for t in reversed(topo_order):
            if t.grad_fn is None:
                continue  # leaf node (no backward op)
            op_cls, ctx, inputs = t.grad_fn
            grad_out = t.grad  # gradient of the output w.rt. this tensor
            # Compute gradients of inputs via this op's backward
            grad_inputs = op_cls.backward(ctx, grad_out)
            if grad_inputs is None:
                grad_inputs = ()
            elif not isinstance(grad_inputs, tuple):
                grad_inputs = (grad_inputs,)
            # Accumulate gradients into input tensors
            for inp, grad in zip(inputs, grad_inputs):
                if isinstance(inp, Tensor) and inp.requires_grad and grad is not None:
                    grad = np.array(grad, dtype=np.float32)  # ensure numpy
                    if inp.grad is None:
                        inp.grad = grad
                    else:
                        inp.grad += grad


def test_binary_op(
    fn_ours,
    fn_torch,
    shape_x,
    shape_y,
):
    x_torch = torch.rand(shape_x).requires_grad_()
    y_torch = torch.rand(shape_y).requires_grad_()
    x_ours = Tensor(x_torch.detach().numpy(), requires_grad=True)
    y_ours = Tensor(y_torch.detach().numpy(), requires_grad=True)

    result_torch = fn_torch(x_torch, y_torch)
    result_ours = fn_ours(x_ours, y_ours)
    grad_outputs = torch.rand(result_torch.shape)
    result_torch.backward(grad_outputs)
    result_ours.backward(grad_outputs.cpu().numpy())

    msg = f"fn={fn_ours} {shape_x} {shape_y}"
    assert_allclose(
        result_ours.data,
        result_torch.detach().cpu().numpy(),
        err_msg=f"output error: {msg}",
        strict=True,
        rtol=1e-6,
    )

    if x_torch.grad is not None:
        assert x_ours.grad is not None
        assert_allclose(
            x_ours.grad,
            x_torch.grad.cpu().numpy(),
            err_msg=f"grad_x error: {msg}",
            strict=True,
            rtol=1e-4,
        )
    if y_torch.grad is not None:
        assert y_ours.grad is not None
        assert_allclose(
            y_ours.grad,
            y_torch.grad.cpu().numpy(),
            err_msg=f"grad_y error: {msg}",
            strict=True,
            rtol=1e-4,
        )


def test_unary_op(fn_ours, fn_torch, shape):
    x_torch = torch.rand(shape).requires_grad_()
    x_ours = Tensor(x_torch.detach().numpy(), requires_grad=True)

    result_torch = fn_torch(x_torch)
    result_ours = fn_ours(x_ours)
    grad_outputs = torch.rand(result_torch.shape)
    result_torch.backward(grad_outputs)
    result_ours.backward(grad_outputs.cpu().numpy())

    msg = f"fn={fn_ours}"
    assert x_torch.grad is not None and x_ours.grad is not None
    assert_allclose(
        result_ours.data,
        result_torch.detach().cpu().numpy(),
        err_msg=f"output error: {msg}",
        strict=True,
        rtol=1e-4,
    )
    assert_allclose(
        x_ours.grad,
        x_torch.grad.cpu().numpy(),
        err_msg=f"grad error: {msg}",
        strict=True,
        rtol=1e-4,
    )


test_unary_op(Sum.apply, torch.sum, (2, 3))
test_unary_op(Sum.apply, torch.sum, (3, 10, 1, 3))
test_unary_op(Sum.apply, torch.sum, (3,))
test_unary_op(
    lambda x: Sum.apply(x, 1, False),
    lambda x: torch.sum(x, 1, False),
    (5, 6, 7),
)
test_unary_op(
    lambda x: Sum.apply(x, (1, 3), False),
    lambda x: torch.sum(x, (1, 3), False),
    (5, 6, 7, 8, 9),
)
test_unary_op(Sum.apply, torch.sum, (3,))
test_unary_op(ReLU.apply, torch.relu, (3,))
test_unary_op(ReLU.apply, torch.relu, (3, 1))
test_unary_op(ReLU.apply, torch.relu, (3, 4, 5, 6))
test_unary_op(Log.apply, torch.log, ())
test_unary_op(Log.apply, torch.log, (3, 4))
test_unary_op(Log.apply, torch.log, (3, 4, 5, 6))
test_unary_op(Softmax.apply, lambda x: torch.softmax(x, dim=-1), (3, 4))
test_unary_op(Softmax.apply, lambda x: torch.softmax(x, dim=-1), ())
test_unary_op(Softmax.apply, lambda x: torch.softmax(x, dim=-1), (10, 10))


test_binary_op(Add.apply, torch.add, (2, 3), (2, 1))
test_binary_op(Add.apply, torch.add, (2, 3), (1,))
test_binary_op(Add.apply, torch.add, (2, 3), (2, 1))
test_binary_op(Mul.apply, torch.mul, (2, 3), (2, 3))
test_binary_op(Mul.apply, torch.mul, (2, 3), (1, 3))
test_binary_op(Mul.apply, torch.mul, (2, 3, 5), (5,))
test_binary_op(MatMul.apply, torch.matmul, (2, 3), (3, 10))
test_binary_op(MatMul.apply, torch.matmul, (2, 1), (1, 10))
test_binary_op(CrossEntropyLoss.apply, torch.nn.CrossEntropyLoss(), (10, 5), (10, 5))
test_binary_op(CrossEntropyLoss.apply, torch.nn.CrossEntropyLoss(), (10,), (10,))


def test_nll_loss(batch, classes):
    labels = torch.randint(0, classes, () if not batch else (batch,))
    onehot = torch.nn.functional.one_hot(labels, num_classes=classes)
    pred = torch.rand(*onehot.shape).requires_grad_()
    pred_ours = Tensor(pred.detach().numpy(), requires_grad=True)
    labels_ours = Tensor(onehot.detach().numpy(), requires_grad=True)

    nll_ours = NLLLoss.apply(pred_ours, labels_ours)
    nll_torch = torch.nn.NLLLoss()(pred, labels)

    nll_ours.backward()
    nll_torch.backward()

    assert_allclose(
        nll_ours.data,
        nll_torch.detach().cpu().numpy(),
        err_msg="output error: NLLLoss",
        strict=True,
        rtol=1e-4,
    )

    assert pred_ours.grad is not None and pred.grad is not None
    assert_allclose(
        pred_ours.grad,
        pred.grad.cpu().numpy(),
        err_msg="output error: NLLLoss",
        strict=True,
        rtol=1e-4,
    )


test_nll_loss(10, 5)
test_nll_loss(10, 10)
test_nll_loss(None, 10)
