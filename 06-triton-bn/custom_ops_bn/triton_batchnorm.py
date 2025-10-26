"""
Triton kernels for batch normalization.

This module provides pure Triton kernel implementations for batch normalization.
The kernels use Welford's online algorithm for numerically stable mean and
variance computation.

For PyTorch integration, use the custom operators registered in ops.py.

Note: For production use, the CUDA implementation is recommended for better
performance and stability.
"""

import triton
import triton.language as tl


@triton.jit
def batchnorm_forward_training_kernel(
    # Pointers
    input_ptr,  # Input tensor pointer
    gamma_ptr,  # Scale parameter pointer
    beta_ptr,  # Shift parameter pointer
    output_ptr,  # Output tensor pointer
    mean_ptr,  # Saved mean pointer
    invstd_ptr,  # Saved inverse std pointer
    running_mean_ptr,  # Running mean pointer (updated in-place)
    running_var_ptr,  # Running variance pointer (updated in-place)
    # Dimensions
    N,
    C,
    spatial_dim,  # Batch size, channels, spatial dimension (H*W)
    # Parameters
    momentum,  # Momentum for running stats
    eps,  # Epsilon for numerical stability
    # Strides
    stride_n,
    stride_c,
    stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)  # One program per channel
    M = N * spatial_dim  # Number of all elements in channel (NHW)

    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    # Welford's algorithm:
    # "Note on a method for calculating corrected sums of squares and products"
    # https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/

    mean = 0.0
    m2 = 0.0
    count = 0

    for m_start in range(0, M, BLOCK_SIZE):  # Over all elements, in blocks
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE)
        mask = m_offsets < M
        n_idxs = m_offsets // spatial_dim  # Index in batch dimension
        s_idxs = m_offsets % spatial_dim  # Index in height/width dimensions
        offset = n_idxs * stride_n + c * stride_c + s_idxs * stride_s

        x = tl.load(input_ptr + offset, mask=mask, other=0.0)

        # Welford's update
        new_count = count + tl.sum(mask.to(tl.int32))
        delta = x - mean
        mean += tl.sum(delta * mask, axis=0) / new_count
        delta2 = x - mean
        m2 += tl.sum(delta * delta2 * mask, axis=0)
        count = new_count

    variance = m2 / count
    invstd = tl.math.rsqrt(variance + eps)

    # EMA updates

    old_running_mean = tl.load(running_mean_ptr + c)
    old_running_var = tl.load(running_var_ptr + c)
    new_running_mean = (1.0 - momentum) * old_running_mean + momentum * mean
    new_running_var = (1.0 - momentum) * old_running_var + momentum * variance
    tl.store(running_mean_ptr + c, new_running_mean)
    tl.store(running_var_ptr + c, new_running_var)

    # Scale/shift and store

    tl.store(mean_ptr + c, mean)
    tl.store(invstd_ptr + c, invstd)

    for m_start in range(0, M, BLOCK_SIZE):
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE)
        mask = m_offsets < M
        n_idxs = m_offsets // spatial_dim
        s_idxs = m_offsets % spatial_dim
        offset = n_idxs * stride_n + c * stride_c + s_idxs * stride_s

        x = tl.load(input_ptr + offset, mask=mask)
        x_normalized = (x - mean) * invstd
        y = x_normalized * gamma + beta
        tl.store(output_ptr + offset, y, mask=mask)


@triton.jit
def batchnorm_forward_inference_kernel(
    # Pointers
    input_ptr,  # Input tensor pointer
    gamma_ptr,  # Scale parameter pointer
    beta_ptr,  # Shift parameter pointer
    mean_ptr,  # Running mean pointer
    var_ptr,  # Running variance pointer
    output_ptr,  # Output tensor pointer
    invstd_ptr,  # Saved inverse std pointer (for backward)
    # Dimensions
    N,
    C,
    spatial_dim,  # Batch size, channels, spatial dimension
    # Parameters
    eps,  # Epsilon for numerical stability
    # Strides
    stride_n,
    stride_c,
    stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    M = N * spatial_dim

    mean = tl.load(mean_ptr + c)
    variance = tl.load(var_ptr + c)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    invstd = tl.math.rsqrt(variance + eps)

    # Scale/shift and store

    for m_start in range(0, M, BLOCK_SIZE):
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE)
        mask = m_offsets < M
        n_idxs = m_offsets // spatial_dim
        s_idxs = m_offsets % spatial_dim
        offset = n_idxs * stride_n + c * stride_c + s_idxs * stride_s

        x = tl.load(input_ptr + offset, mask=mask)
        x_normalized = (x - mean) * invstd
        y = x_normalized * gamma + beta
        tl.store(output_ptr + offset, y, mask=mask)


@triton.jit
def batchnorm_backward_kernel(
    # Pointers
    grad_output_ptr,  # Gradient from next layer
    input_ptr,  # Original input
    gamma_ptr,  # Scale parameter
    mean_ptr,  # Saved mean from forward
    invstd_ptr,  # Saved inverse std from forward
    grad_input_ptr,  # Gradient w.r.t. input (output)
    grad_gamma_ptr,  # Gradient w.r.t. gamma (output)
    grad_beta_ptr,  # Gradient w.r.t. beta (output)
    # Dimensions
    N,
    C,
    spatial_dim,
    # Strides
    stride_n,
    stride_c,
    stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    M = N * spatial_dim

    mean = tl.load(mean_ptr + c)
    invstd = tl.load(invstd_ptr + c)
    gamma = tl.load(gamma_ptr + c)

    # Get sums for gradients:
    # dL/dbeta = sum(dL/dy)
    # dL/dgamma = sum(dL/dy * xhat)

    sum_dy = 0.0
    sum_dy_x_hat = 0.0

    for m_start in range(0, M, BLOCK_SIZE):
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE)
        mask = m_offsets < M
        n_idxs = m_offsets // spatial_dim
        s_idxs = m_offsets % spatial_dim
        offset = n_idxs * stride_n + c * stride_c + s_idxs * stride_s

        x = tl.load(input_ptr + offset, mask=mask, other=0.0)
        dy = tl.load(grad_output_ptr + offset, mask=mask, other=0.0)
        x_hat = (x - mean) * invstd
        sum_dy += tl.sum(dy, axis=0)
        sum_dy_x_hat += tl.sum(dy * x_hat, axis=0)

    # Final gradients

    tl.store(grad_beta_ptr + c, sum_dy)
    tl.store(grad_gamma_ptr + c, sum_dy_x_hat)

    for m_start in range(0, M, BLOCK_SIZE):
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE)
        mask = m_offsets < M
        n_idxs = m_offsets // spatial_dim
        s_idxs = m_offsets % spatial_dim
        offset = n_idxs * stride_n + c * stride_c + s_idxs * stride_s

        x = tl.load(input_ptr + offset, mask=mask)
        dy = tl.load(grad_output_ptr + offset, mask=mask)
        x_hat = (x - mean) * invstd
        w1 = M * dy
        w2 = sum_dy
        w3 = x_hat * sum_dy_x_hat
        dx = (gamma * invstd / M) * (w1 - w2 - w3)
        tl.store(grad_input_ptr + offset, dx, mask=mask)
