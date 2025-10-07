import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("pgf")
plt.rcParams.update({"font.family": "serif", "pgf.rcfonts": False})

data_ours = {
    1: {"DP": 47.905, "DDP": 46.740, "DDP+DALI": 22.937},
    2: {"DP": 42.863, "DDP": 26.947, "DDP+DALI": 13.018},
}


data_baseline = {
    1: {"DP": 30.398, "DDP": 32.615, "DDP+DALI": 10.646},
    2: {"DP": 32.169, "DDP": 17.647, "DDP+DALI": 7.195},
    4: {"DP": 34.362, "DDP": 13.558, "DDP+DALI": 6.722},
}


def main(data):
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), layout="constrained")

    x = np.arange(3)
    width = 0.25
    sep = 0.03

    for i, (label, y) in enumerate(data.items()):
        offset = (width + sep) * i
        rects = ax.bar(
            x + offset,
            list(y.values()),
            width,
            color=("tab:blue", 0.2 + 0.8 * i / (len(data) - 1)),
            label=label,
            edgecolor="black",
        )
        ax.bar_label(rects, padding=3, fmt="{:.1f}")

    ax.set_ylabel("Training Time (s) $\\downarrow$")
    ax.set_xticks(x + (width + sep) / 2, data[1].keys())
    ax.legend(title="GPU Count", ncols=1)
    ax.set_ylim(0, 55)

    fig.savefig("plot.pdf", backend="pgf", dpi=300)


if __name__ == "__main__":
    main(data_ours)
