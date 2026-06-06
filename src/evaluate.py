from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


BASELINE_MODEL = "openai-community/gpt2"
HUMAN_COLUMNS = [
    "model_type",
    "decoding",
    "prompt",
    "rater",
    "fluency",
    "coherence",
    "creativity",
    "emotional_tone",
    "lyric_like_quality",
]


class LyricsDataset(Dataset):
    def __init__(self, text_path: Path, tokenizer, block_size: int, max_samples: int | None = None) -> None:
        if not text_path.exists():
            raise FileNotFoundError(f"Missing evaluation data file: {text_path}")
        lines = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_samples:
            lines = lines[:max_samples]
        self.examples = [
            tokenizer(line, truncation=True, max_length=block_size, padding=False)["input_ids"]
            for line in lines
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> list[int]:
        return self.examples[index]


def collate_batch(batch: list[list[int]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item) for item in batch)
    input_ids = []
    attention_mask = []
    for item in batch:
        padding = [pad_token_id] * (max_len - len(item))
        input_ids.append(item + padding)
        attention_mask.append([1] * len(item) + [0] * len(padding))
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    labels = input_tensor.clone()
    labels[mask_tensor == 0] = -100
    return {"input_ids": input_tensor, "attention_mask": mask_tensor, "labels": labels}


@torch.no_grad()
def compute_loss(model_path: str | Path, data_path: Path, block_size: int, batch_size: int, max_samples: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()

    dataset = LyricsDataset(data_path, tokenizer, block_size, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))

    loss = sum(losses) / max(1, len(losses))
    return {
        "model_path": str(model_path),
        "examples": len(dataset),
        "loss": loss,
        "perplexity": math.exp(min(loss, 20)),
    }


def latest_samples_file(samples_dir: Path) -> Path | None:
    files = sorted(samples_dir.glob("generated_samples_*.csv"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def create_human_eval_template(output_dir: Path, samples_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = output_dir / "human_eval_template.csv"
    scores_path = output_dir / "human_eval_scores.csv"
    samples_path = latest_samples_file(samples_dir)

    rows = []
    if samples_path:
        samples = pd.read_csv(samples_path)
        for _, row in samples.iterrows():
            rows.append(
                {
                    "model_type": row.get("model_type", ""),
                    "decoding": row.get("decoding", ""),
                    "prompt": row.get("prompt", ""),
                    "rater": "",
                    "fluency": "",
                    "coherence": "",
                    "creativity": "",
                    "emotional_tone": "",
                    "lyric_like_quality": "",
                }
            )

    if not rows:
        rows = [{column: "" for column in HUMAN_COLUMNS}]

    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if not scores_path.exists():
        with scores_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HUMAN_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return template_path


def summarize_human_eval(output_dir: Path) -> dict:
    scores_path = output_dir / "human_eval_scores.csv"
    if not scores_path.exists():
        return {
            "status": "missing_scores",
            "message": "Run evaluation once to create outputs/eval/human_eval_scores.csv.",
        }

    df = pd.read_csv(scores_path)
    rating_columns = ["fluency", "coherence", "creativity", "emotional_tone", "lyric_like_quality"]
    for column in rating_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[rating_columns].dropna(how="all").empty:
        return {
            "status": "empty_scores",
            "message": "Fill 1-5 ratings in outputs/eval/human_eval_scores.csv to compute a summary.",
        }

    summary = (
        df.groupby(["model_type", "decoding"], dropna=False)[rating_columns]
        .mean()
        .round(3)
        .reset_index()
    )
    summary_path = output_dir / "human_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {"status": "summarized", "summary_path": str(summary_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 lyrics models.")
    parser.add_argument("--baseline_model", default=BASELINE_MODEL)
    parser.add_argument("--model_dir", type=Path, default=Path("models") / "gpt2-lyrics")
    parser.add_argument("--data_dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--samples_dir", type=Path, default=Path("outputs") / "samples")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs") / "eval")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_eval_samples", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_path = args.data_dir / "test.txt"

    results = [
        {
            "model_type": "baseline",
            **compute_loss(args.baseline_model, test_path, args.block_size, args.batch_size, args.max_eval_samples),
        }
    ]

    if args.model_dir.exists() and (args.model_dir / "config.json").exists():
        results.append(
            {
                "model_type": "fine_tuned",
                **compute_loss(args.model_dir, test_path, args.block_size, args.batch_size, args.max_eval_samples),
            }
        )
    else:
        print(f"Fine-tuned model not found at {args.model_dir}; evaluating baseline only.")

    metrics_path = args.output_dir / "perplexity_results.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    (args.output_dir / "perplexity_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    template_path = create_human_eval_template(args.output_dir, args.samples_dir)
    human_summary = summarize_human_eval(args.output_dir)
    summary = {
        "perplexity_results": str(metrics_path),
        "human_eval_template": str(template_path),
        "human_eval": human_summary,
    }
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
