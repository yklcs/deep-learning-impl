import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm

matplotlib.use("pgf")
plt.rcParams.update({"font.family": "serif", "pgf.rcfonts": False})


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_custom_ops_bn import BatchNormCustom


def test_custom_batchnorm(seed):
    torch.manual_seed(seed)

    # test configuration
    N, C, H, W = 32, 64, 56, 56
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create input tensor [N, C, H, W]
    input_custom = torch.randn(N, C, H, W, device=device, requires_grad=True)
    input_pytorch = input_custom.clone().detach().requires_grad_(True)

    # Initialize parameters
    gamma = torch.ones(C, device=device, requires_grad=True)
    beta = torch.zeros(C, device=device, requires_grad=True)
    running_mean = torch.zeros(C, device=device)
    running_var = torch.ones(C, device=device)

    # PyTorch's BatchNorm
    bn_pytorch = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1, device=device)
    bn_pytorch.weight.data = gamma.clone()
    bn_pytorch.bias.data = beta.clone()
    bn_pytorch.running_mean.data = running_mean.clone()
    bn_pytorch.running_var.data = running_var.clone()

    # Forward pass ************************************************************************************************************
    output_custom = BatchNormCustom.apply(
        input_custom, gamma, beta, running_mean, running_var, True, 0.1, 1e-5
    )
    output_pytorch = bn_pytorch(input_pytorch)

    forward_diff = torch.abs(output_custom - output_pytorch).max().item()

    # Backward pass
    grad_output = torch.randn_like(output_custom)

    output_custom.backward(grad_output)
    output_pytorch.backward(grad_output.clone())

    input_grad_diff = torch.abs(input_custom.grad - input_pytorch.grad).max().item()
    gamma_grad_diff = torch.abs(gamma.grad - bn_pytorch.weight.grad).max().item()
    beta_grad_diff = torch.abs(beta.grad - bn_pytorch.bias.grad).max().item()

    mean_diff = torch.abs(running_mean - bn_pytorch.running_mean).max().item()
    var_diff = torch.abs(running_var - bn_pytorch.running_var).max().item()

    return {
        "Training Output": forward_diff,
        "Input Grad.": input_grad_diff,
        "Gamma Grad.": gamma_grad_diff,
        "Beta Grad.": beta_grad_diff,
        "Mean": mean_diff,
        "Variance": var_diff,
    }


def test_inference_mode(seed, thresh=1e-5):
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # inference configuration
    N, C, H, W = 8, 128, 28, 28

    # Create test data
    input_tensor = torch.randn(N, C, H, W, device=device)
    gamma = torch.ones(C, device=device)
    beta = torch.zeros(C, device=device)
    running_mean = torch.randn(C, device=device) * 0.1
    running_var = torch.abs(torch.randn(C, device=device)) * 0.5 + 0.8

    output_custom = BatchNormCustom.apply(
        input_tensor,
        gamma,
        beta,
        running_mean.clone(),
        running_var.clone(),
        False,
        0.1,
        1e-5,
    )

    bn_pytorch = nn.BatchNorm2d(C, eps=1e-5, device=device)
    bn_pytorch.weight.data = gamma
    bn_pytorch.bias.data = beta
    bn_pytorch.running_mean.data = running_mean.clone()
    bn_pytorch.running_var.data = running_var.clone()
    bn_pytorch.eval()

    output_pytorch = bn_pytorch(input_tensor)

    diff = torch.abs(output_custom - output_pytorch).max().item()

    return {"Inference Output": diff}


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(7, 2), layout="constrained")

    results = {}
    trials = 500
    for i in tqdm(range(trials)):
        training = test_custom_batchnorm(i)
        inference = test_inference_mode(i)
        result = training
        for k, v in result.items():
            if k in results:
                results[k].append(v)
            else:
                results[k] = [v]

    sns.stripplot(
        data=results,
        ax=ax,
        orient="h",
        alpha=0.3,
        size=3,
        jitter=0.2,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Errors")
    fig.savefig("diffs.pdf", dpi=300)
