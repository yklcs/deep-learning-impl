import json
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

timings = {}


def load(name):
    with open(f"{name}.json", "r") as f:
        timings[name] = json.load(f)


load("onnx-timings-quantized-jit")
load("onnx-timings-quantized-nojit")
load("onnx-timings-noquantized-jit")
load("onnx-timings-noquantized-nojit")

load("pytorch-timings-quantized-jit")
load("pytorch-timings-quantized-nojit")
load("pytorch-timings-noquantized-jit")
load("pytorch-timings-noquantized-nojit")


def get_title(s: str):
    title = ""
    if "onnx" in s:
        title += r"ONNX"
    elif "pytorch" in s:
        title += "PyTorch"

    if "-quantized" in s:
        title += "\nQuantized"
    else:
        title += "\nNo Quantization"

    if "-jit" in s:
        title += "\nJIT"
    else:
        title += "\nNo JIT"

    return title


nbins = 1000
bins = np.linspace(
    # min(t for ts in timings.values() for t in ts),
    0,
    # max(t for ts in timings.values() for t in ts),
    0.125,
    nbins,
)

fig, axs = plt.subplots(
    nrows=8,
    ncols=1,
    sharex=True,
    # sharey=True,
    figsize=(10, 8),
    layout="constrained",
)

colors = matplotlib.color_sequences["tab10"]

for i, (name, timing) in enumerate(timings.items()):
    ax = cast(Axes, axs[i])

    ax.hist(
        timing,
        bins=bins,
        label=name,
        color=colors[3] if "pytorch" in name else colors[0],
    )
    # ax.set_xscale("log")
    mean = np.mean(timing)
    ax.text(
        0.99,
        0.9,
        get_title(name),
        ha="right",
        va="top",
        transform=ax.transAxes,
    )
    ax.axvline(mean, c="black")
    # ax.legend()
fig.supxlabel("Inference time (seconds, lower is better)")

fig.savefig("timings.png", dpi=300, bbox_inches="tight")
