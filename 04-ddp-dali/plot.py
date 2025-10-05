import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("pgf")
plt.rcParams.update({"font.family": "serif", "pgf.rcfonts": False})

data = {
    1: {"DP": 48.508, "DDP": 47.083, "DDP+DALI": 22.700},
    2: {"DP": 42.485, "DDP": 24.431, "DDP+DALI": 11.891},
}


def main():
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), layout="constrained")

    x = np.arange(3)
    width = 0.25
    sep = 0.03

    for i, (label, y) in enumerate(data.items()):
        offset = (width + sep) * i
        rects = ax.bar(
            x + offset,
            list(y.values()),
            width,
            color=("tab:blue", 0.5 + 0.5 * i / (len(data) - 1)),
            label=label,
            edgecolor="black",
        )
        ax.bar_label(rects, padding=3)

    ax.set_ylabel("Training Time (s)")
    ax.set_title("Training Time, 1 Epoch (Our Measurements)")
    ax.set_xticks(x + 0.5 * width + sep / 2, data[1].keys())
    ax.legend(title="GPU Count", ncols=2)
    ax.set_ylim(0, 55)

    fig.savefig("plot.pdf", backend="pgf", dpi=300)


if __name__ == "__main__":
    main()
