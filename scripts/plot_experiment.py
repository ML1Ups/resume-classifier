from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DIRECTORY = Path("docs/experiment")
PANELS = [
    ("Queue depth, messages", ["ready", "unacked"]),
    ("Source rate and throughput, tasks/s", ["done_rate"]),
    ("Task time, s", ["latency_seconds"]),
    ("Memory, MiB", ["worker_mb", "rabbitmq_mb", "source_mb"]),
]


def main() -> None:
    runs = {mode: pd.read_csv(DIRECTORY / f"{mode}.csv") for mode in ("limited", "unlimited")}
    sources = {mode: pd.read_csv(DIRECTORY / f"{mode}_source.csv") for mode in runs}
    fig, axes = plt.subplots(len(PANELS), 2, figsize=(13, 16), sharex=True)
    for column, (mode, frame) in enumerate(runs.items()):
        scaled = frame.loc[frame["workers"] == 2, "seconds"].min()
        for row, (title, metrics) in enumerate(PANELS):
            ax = axes[row][column]
            for metric in metrics:
                ax.plot(frame["seconds"], frame[metric], label=metric)
            if metrics == ["done_rate"]:
                source = sources[mode]
                ax.plot(source["seconds"], source["rate"], label="source rate")
                ax.set_yscale("log")
            ax.axvline(scaled, color="grey", linestyle="--", label="second worker")
            ax.set_title(f"{mode}: {title}")
            ax.legend(fontsize=8)
        axes[-1][column].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(DIRECTORY / "experiment.png", dpi=110)


if __name__ == "__main__":
    main()
