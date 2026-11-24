"""Check the saved Colab results and regenerate the public table and chart."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEEDS = (1776, 2027, 31415)
MODELS = ("shallow_wide", "deep_narrow")
BUDGETS = ("equal_bytes", "equal_time")
PARAMETERS = dict(zip(MODELS, (357024, 361408)))
BYTE_TARGET = 10567680
TIME_TARGET = 1800
MATCH_KEYS = (
    "runtime_id", "cpu", "threads", "torch", "revision", "train_sha256",
    "eval_sha256", "eval_bytes", "target_bytes", "time_budget_s",
)


def load_results():
    lines = (RESULTS / "colab-output.log").read_text(encoding="utf-8").splitlines()

    def records(prefix):
        return [json.loads(line[len(prefix):]) for line in lines if line.startswith(prefix)]

    individual = records("RESULT ")
    runs = {(r["seed"], r["model"]): r for r in records("RUN ")}
    rows = {(r["seed"], r["model"], r["checkpoint"]): r for r in individual}
    pairs = {p["seed"]: p for p in records("PAIR RESULT ")}
    expected = {(seed, model, budget) for seed in SEEDS for model in MODELS for budget in BUDGETS}
    if len(individual) != len(expected) or rows.keys() != expected or set(pairs) != set(SEEDS):
        raise ValueError("Missing or duplicate measurements")

    first = runs[SEEDS[0], MODELS[0]]
    for seed in SEEDS:
        pair = pairs[seed]
        shallow, deep = (runs[seed, model] for model in MODELS)
        if any(shallow[key] != deep[key] for key in MATCH_KEYS):
            raise ValueError(f"Unmatched runtime or protocol for seed {seed}")
        if shallow["runtime_id"] != pair["runtime_id"] or shallow["threads"] != 2:
            raise ValueError(f"Unmatched runtime ID or thread count for seed {seed}")
        if any(shallow[key] != first[key] for key in MATCH_KEYS[3:]):
            raise ValueError(f"Protocol or dataset mismatch for seed {seed}")

    for (seed, model, budget), row in rows.items():
        if row != pairs[seed]["models"][model][budget] or row["bytes"] != 1024 * row["steps"]:
            raise ValueError(f"Inconsistent saved measurement: {row}")
        if budget == "equal_bytes" and row["bytes"] != BYTE_TARGET:
            raise ValueError(f"Invalid equal-byte checkpoint: {row}")
        if budget == "equal_time" and not TIME_TARGET <= row["training_seconds"] < TIME_TARGET + 1:
            raise ValueError(f"Unexpected time checkpoint in the saved run: {row}")
    return rows, runs


def write_csv(rows, runs):
    with (RESULTS / "measurements.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow([
            "seed", "model", "parameters", "checkpoint", "steps", "training_bytes",
            "training_seconds", "validation_bits_per_byte", "bytes_per_training_second",
            "cpu", "torch_threads", "torch", "runtime_id",
        ])
        for (seed, model, budget), r in sorted(rows.items()):
            run = runs[seed, model]
            writer.writerow([
                seed, model, PARAMETERS[model], budget, r["steps"], r["bytes"],
                r["training_seconds"], r["bits_per_byte"],
                r["bytes"] / r["training_seconds"],
                run["cpu"], run["threads"], run["torch"], run["runtime_id"],
            ])


def means_for(rows, budget):
    return [
        mean(rows[seed, model, budget]["bits_per_byte"] for seed in SEEDS)
        for model in MODELS
    ]


def draw_chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, teal, blue = "#172B3A", "#485E68", "#006C61", "#304D76"
    paper, grid = "#FAFBFA", "#D9E4E6"
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.0), facecolor=paper, sharex=True)
    fig.subplots_adjust(left=.15, right=.92, top=.71, bottom=.145, hspace=.52)
    fig.text(.08, .933, "T i n y S t o r i e s  /  C P U  B E N C H M A R K", color=teal,
             fontsize=10.5, weight="bold")
    fig.text(.08, .869, "Shallow vs. deep Transformers", color=ink, fontsize=25, weight="bold")
    fig.text(.08, .821, "Shallow/wide had lower validation loss in every pair.",
             color=ink, fontsize=14)
    handles = [
        plt.Line2D([], [], color=teal, marker="o", markersize=9, linestyle="",
                   label="Shallow/wide"),
        plt.Line2D([], [], color=blue, marker="s", markersize=8, linestyle="",
                   label="Deep/narrow"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.073, .797),
               ncol=2, frameon=False, fontsize=11.5, labelcolor=ink)

    for ax, budget, heading in zip(
        axes, BUDGETS, ("Same training bytes · 10.57M per model", "Same training time · 30 minutes per model")
    ):
        ax.set_facecolor(paper)
        ax.set_title(heading, fontsize=13.5, color=ink, loc="left", weight="bold", pad=12)
        ax.set_xlim(1.30, 1.59)
        ax.set_ylim(-.28, 2.56)
        ax.set_yticks([2, 1, 0], [str(seed) for seed in SEEDS], color=muted, fontsize=11)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.set_xticks([1.30, 1.35, 1.40, 1.45, 1.50, 1.55])
        ax.grid(axis="x", color=grid)
        ax.set_axisbelow(True)
        for side in ("left", "top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)

        for y, seed in zip((2, 1, 0), SEEDS):
            sw, dn = [rows[seed, model, budget]["bits_per_byte"] for model in MODELS]
            ax.plot([sw, dn], [y, y], linewidth=2, color="#A6B9BE")
            for value, marker, color in ((sw, "o", teal), (dn, "s", blue)):
                ax.scatter(value, y, marker=marker, s=64, color=color, zorder=3)
                ax.annotate(f"{value:.3f}", (value, y), xytext=(0, 9),
                            textcoords="offset points", ha="center", color=color,
                            fontsize=10.5, weight="bold")

    axes[-1].tick_params(axis="x", colors=muted, labelsize=10)
    axes[-1].set_xlabel("Validation bits per byte  ·  Lower is better", color=ink,
                        fontsize=11.5, labelpad=9)
    fig.text(.08, .058, "Rows show paired seeds. Each pair shared a two-thread Colab CPU runtime.",
             color=muted, fontsize=9.7)
    fig.savefig(path, dpi=170, facecolor=paper)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart", action="store_true", help="also regenerate the PNG using matplotlib")
    args = parser.parse_args()
    rows, runs = load_results()
    write_csv(rows, runs)
    for budget in BUDGETS:
        shallow, deep = means_for(rows, budget)
        print(f"{budget}: shallow {shallow:.6f}, deep {deep:.6f}, "
              f"relative mean loss reduction {100 * (1 - shallow / deep):.2f}%")
    print(f"Verified {len(rows)} saved measurements across {len(SEEDS)} matched pairs.")
    if args.chart:
        draw_chart(rows, RESULTS / "TinyStories_CPU_Results.png")


if __name__ == "__main__":
    main()

