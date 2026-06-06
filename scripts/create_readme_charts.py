from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)


def create_perplexity_chart() -> None:
    results = pd.read_csv(ROOT / "results" / "evaluation" / "perplexity_results.csv")
    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    bars = axis.bar(
        ["Baseline GPT-2", "Fine-tuned GPT-2"],
        results["perplexity"],
        color=["#697386", "#167d9a"],
        width=0.58,
    )
    axis.set_ylabel("Test perplexity (lower is better)")
    axis.set_title("Fine-tuning reduced perplexity by 37.2%")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, results["perplexity"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.8,
            f"{value:.2f}",
            ha="center",
            fontweight="bold",
        )
    figure.tight_layout()
    figure.savefig(ASSETS / "perplexity_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_qualitative_chart() -> None:
    results = pd.read_csv(ROOT / "results" / "evaluation" / "human_eval_summary.csv")
    criteria = ["fluency", "coherence", "creativity", "emotional_tone", "lyric_like_quality"]
    results["overall"] = results[criteria].mean(axis=1)
    labels = [f"{row.model_type}\n{row.decoding}" for row in results.itertuples()]

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    bars = axis.bar(
        labels,
        results["overall"],
        color=["#697386", "#bb6b2c", "#167d9a", "#4f8a58"],
        width=0.62,
    )
    axis.set_ylim(0, 5)
    axis.set_ylabel("Mean qualitative score (1-5)")
    axis.set_title("Top-k produced the strongest qualitative result")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, results["overall"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.2f}",
            ha="center",
            fontweight="bold",
        )
    figure.tight_layout()
    figure.savefig(ASSETS / "qualitative_scores.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    create_perplexity_chart()
    create_qualitative_chart()
