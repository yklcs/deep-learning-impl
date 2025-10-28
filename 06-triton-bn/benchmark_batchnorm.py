import time

import matplotlib
import torch
from custom_ops_bn.ops import batchnorm
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

matplotlib.use("pgf")
plt.rcParams.update({"font.family": "serif", "pgf.rcfonts": False})


def batchnorm_cuda(x, g, b, rm, rv, training):
    return batchnorm(x, g, b, rm, rv, training, backend="cuda")


def batchnorm_triton(x, g, b, rm, rv, training):
    return batchnorm(x, g, b, rm, rv, training, backend="triton")


def batchnorm_pytorch(x, g, b, rm, rv, training):
    return nn.functional.batch_norm(
        x,
        running_mean=rm,
        running_var=rv,
        weight=g,
        bias=b,
        training=training,
        momentum=0.1,
        eps=1e-5,
    )


def benchmark_forward_backward(
    func,
    input,
    gamma,
    beta,
    running_mean,
    running_var,
    training=True,
    warmup=50,
    iterations=500,
):
    """Benchmark forward and backward separately"""
    # Warmup
    for _ in range(warmup):
        rm = running_mean.clone()
        rv = running_var.clone()
        inp = input.clone().detach().requires_grad_(True)
        g = gamma.clone().detach().requires_grad_(True)
        b = beta.clone().detach().requires_grad_(True)

        output = func(inp, g, b, rm, rv, training=training)
        loss = output.sum()
        loss.backward()

    torch.cuda.synchronize()

    # Benchmark - measure forward and backward separately
    total_forward_time = 0
    total_backward_time = 0

    for _ in range(iterations):
        rm = running_mean.clone()
        rv = running_var.clone()
        inp = input.clone().detach().requires_grad_(True)
        g = gamma.clone().detach().requires_grad_(True)
        b = beta.clone().detach().requires_grad_(True)

        # Measure forward
        torch.cuda.synchronize()
        start = time.time()
        output = func(inp, g, b, rm, rv, training=training)
        torch.cuda.synchronize()
        total_forward_time += time.time() - start

        # Measure backward
        torch.cuda.synchronize()
        start = time.time()
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()
        total_backward_time += time.time() - start

    avg_forward_time = total_forward_time / iterations * 1000  # Convert to ms
    avg_backward_time = total_backward_time / iterations * 1000  # Convert to ms

    return avg_forward_time, avg_backward_time


def run_benchmark(N, C, H, W, training=True):
    device = "cuda"

    # Create base tensors
    input_base = torch.randn(N, C, H, W, device=device)
    gamma = torch.randn(C, device=device)
    beta = torch.randn(C, device=device)
    running_mean = torch.randn(C, device=device)
    running_var = torch.abs(torch.randn(C, device=device))

    # CUDA custom operator
    forward_time_cuda, backward_time_cuda = benchmark_forward_backward(
        batchnorm_cuda, input_base, gamma, beta, running_mean, running_var, training
    )

    # Triton custom operator
    forward_time_triton, backward_time_triton = benchmark_forward_backward(
        batchnorm_triton,
        input_base,
        gamma,
        beta,
        running_mean,
        running_var,
        training,
    )

    forward_time_pytorch, backward_time_pytorch = benchmark_forward_backward(
        batchnorm_pytorch,
        input_base,
        gamma,
        beta,
        running_mean,
        running_var,
        training,
    )

    return (
        (forward_time_cuda, backward_time_cuda),
        (forward_time_triton, backward_time_triton),
        (forward_time_pytorch, backward_time_pytorch),
    )


def main():
    xs = range(1, 1 + 256, 16)
    # C = 32
    N = 32
    H = W = 56
    configs = [(N, x, H, W) for x in xs]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), layout="constrained")

    fwd_cudas = []
    bwd_cudas = []
    fwd_ourses = []
    bwd_ourses = []
    fwd_pytorchs = []
    bwd_pytorchs = []

    for N, C, H, W in tqdm(configs):
        (
            (fwd_cuda, bwd_cuda),
            (fwd_ours, bwd_ours),
            (
                fwd_pytorch,
                bwd_pytorch,
            ),
        ) = run_benchmark(N, C, H, W, training=True)
        fwd_cudas.append(fwd_cuda)
        bwd_cudas.append(bwd_cuda)
        fwd_ourses.append(fwd_ours)
        bwd_ourses.append(bwd_ours)
        fwd_pytorchs.append(fwd_pytorch)
        bwd_pytorchs.append(bwd_pytorch)

    markersize = 4
    ax.set_title(f"BatchNorm2d Performance ($N = {N}, {H} \\times {W}$)")
    ax.plot(xs, fwd_cudas, "gx:", label="CUDA Forward", markersize=markersize)
    ax.plot(xs, bwd_cudas, "gx-", label="CUDA Backward", markersize=markersize)
    ax.plot(xs, fwd_ourses, "b+:", label="Triton Forward", markersize=markersize)
    ax.plot(xs, bwd_ourses, "b+-", label="Triton Backward", markersize=markersize)
    ax.plot(xs, fwd_pytorchs, "r^:", label="PyTorch Forward", markersize=markersize)
    ax.plot(xs, bwd_pytorchs, "r^-", label="PyTorch Backward", markersize=markersize)
    ax.set_xlabel("Channel Count")
    ax.set_ylabel("Time (ms)")

    ax.legend()

    fig.savefig("triton-bn-bench.pdf", dpi=300)


if __name__ == "__main__":
    main()
