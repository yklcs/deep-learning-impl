import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = """
Model             PagedAttn   Batch   Latency     Throughput          TTFT        TPOT
Llama-2-7b-hf     ON          1       3.92        65.4                22.04       15.26       
Llama-2-7b-hf     OFF         1       3.91        65.5                21.06       15.23       
Llama-2-7b-hf     ON          2       4.00        127.9               29.59       15.55       
Llama-2-7b-hf     OFF         2       4.00        128.0               28.68       15.53       
Llama-2-7b-hf     ON          4       4.04        253.2               33.21       15.70       
Llama-2-7b-hf     OFF         4       4.05        252.7               33.36       15.73       
Llama-2-7b-hf     ON          8       4.18        490.4               36.39       16.21       
Llama-2-7b-hf     OFF         8       8.07        253.8               2044.36     15.71       
Llama-2-7b-hf     ON          32      5.00        1636.8              52.89       19.38       
Llama-2-7b-hf     OFF         32      32.22       254.2               14117.16    15.72       
Llama-2-7b-hf     ON          64      7.38        2221.1              74.47       23.51       
Llama-2-7b-hf     OFF         64      64.43       254.3               30227.72    15.72       
Meta-Llama-3-8B   ON          1       4.44        57.7                23.51       17.31       
Meta-Llama-3-8B   OFF         1       4.44        57.7                22.86       17.31       
Meta-Llama-3-8B   ON          2       4.48        114.3               31.15       17.41       
Meta-Llama-3-8B   OFF         2       4.48        114.2               31.33       17.41       
Meta-Llama-3-8B   ON          4       4.50        227.5               36.26       17.48       
Meta-Llama-3-8B   OFF         4       4.49        228.1               35.55       17.44       
Meta-Llama-3-8B   ON          8       4.55        450.3               38.96       17.65       
Meta-Llama-3-8B   OFF         8       4.55        450.6               39.06       17.64       
Meta-Llama-3-8B   ON          32      5.10        1607.7              56.91       19.71       
Meta-Llama-3-8B   OFF         32      18.19       450.4               6021.84     17.76       
Meta-Llama-3-8B   ON          64      5.68        2885.0              127.17      21.85       
Meta-Llama-3-8B   OFF         64      36.41       450.0               14015.08    17.82    
"""

df = pd.read_csv(io.StringIO(data), sep=r"\s+")

mpl.use("pgf")
plt.rcParams.update({"font.family": "serif", "pgf.rcfonts": False})

fig, ax = plt.subplots(1, 1, figsize=(8, 4), layout="constrained")


ax.plot(
    df[(df["Model"] == "Llama-2-7b-hf") & (df["PagedAttn"] == "ON")]["Batch"],
    df[(df["Model"] == "Llama-2-7b-hf") & (df["PagedAttn"] == "ON")]["Throughput"],
    label="Llama 2, PagedAttn",
    marker="o",
    color="green",
    linestyle="-",
)

ax.plot(
    df[(df["Model"] == "Llama-2-7b-hf") & (df["PagedAttn"] == "OFF")]["Batch"],
    df[(df["Model"] == "Llama-2-7b-hf") & (df["PagedAttn"] == "OFF")]["Throughput"],
    label="Llama 2, No PagedAttn",
    marker="x",
    color="green",
    linestyle="-",
)

ax.plot(
    df[(df["Model"] == "Meta-Llama-3-8B") & (df["PagedAttn"] == "ON")]["Batch"],
    df[(df["Model"] == "Meta-Llama-3-8B") & (df["PagedAttn"] == "ON")]["Throughput"],
    label="Llama 3, PagedAttn",
    marker="o",
    color="purple",
    linestyle="--",
)

ax.plot(
    df[(df["Model"] == "Meta-Llama-3-8B") & (df["PagedAttn"] == "OFF")]["Batch"],
    df[(df["Model"] == "Meta-Llama-3-8B") & (df["PagedAttn"] == "OFF")]["Throughput"],
    label="Llama 3, No PagedAttn",
    marker="x",
    color="purple",
    linestyle="--",
)


ax.set_xlabel("Batch Size")
ax.set_ylabel("Throughput (tok/s)")

ax.legend()

fig.savefig("plot.png", dpi=300, bbox_inches="tight")
