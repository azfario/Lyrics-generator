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

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


DEFAULT_MODEL_NAME = "openai-community/gpt2"


class LyricsDataset(Dataset):
    def __init__(
        self,
        text_path: Path,
        tokenizer,
        block_size: int,
        max_samples: int | None = None,
    ) -> None:
        if not text_path.exists():
            raise FileNotFoundError(f"Missing data file: {text_path}")

        lines = [line.strip() for line in text_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_samples:
            lines = lines[:max_samples]
        if not lines:
            raise ValueError(f"No training examples found in {text_path}")

        self.examples = [
            tokenizer(
                line,
                truncation=True,
                max_length=block_size,
                padding=False,
            )["input_ids"]
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
def evaluate_loss(model, loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_log_row(log_path: Path, row: dict[str, float | int | str]) -> None:
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on song lyrics.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data_dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--output_dir", type=Path, default=Path("models") / "gpt2-lyrics")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=256)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.output_dir = Path("models") / "gpt2-lyrics-smoke"
        args.epochs = 1
        args.batch_size = 2
        args.gradient_accumulation_steps = 2
        args.max_steps = 5
        args.max_train_samples = 32
        args.max_val_samples = 16
        args.eval_every = 5
        args.save_every = 5

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    max_train_samples = args.max_train_samples or None
    max_val_samples = args.max_val_samples or None
    train_dataset = LyricsDataset(args.data_dir / "train.txt", tokenizer, args.block_size, max_train_samples)
    val_dataset = LyricsDataset(args.data_dir / "validation.txt", tokenizer, args.block_size, max_val_samples)

    collate = lambda batch: collate_batch(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_update_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    if args.max_steps:
        total_update_steps = min(total_update_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(0, total_update_steps // 10)),
        num_training_steps=max(1, total_update_steps),
    )

    use_fp16 = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    log_path = args.output_dir / "training_log.csv"
    best_val_loss = float("inf")
    global_step = 0
    running_loss = 0.0
    running_loss_count = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            running_loss_count += 1

            should_step = batch_index % args.gradient_accumulation_steps == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.eval_every == 0 or (args.max_steps and global_step >= args.max_steps):
                    train_loss = running_loss / max(1, running_loss_count)
                    running_loss = 0.0
                    running_loss_count = 0
                    val_loss = evaluate_loss(model, val_loader, device)
                    val_ppl = math.exp(min(val_loss, 20))
                    row = {
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": round(train_loss, 6),
                        "validation_loss": round(val_loss, 6),
                        "validation_perplexity": round(val_ppl, 6),
                    }
                    save_log_row(log_path, row)
                    print(json.dumps(row))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = args.output_dir / "best"
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)

                if global_step % args.save_every == 0:
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

                if args.max_steps and global_step >= args.max_steps:
                    break

        if args.max_steps and global_step >= args.max_steps:
            break

    final_val_loss = evaluate_loss(model, val_loader, device)
    final_metrics = {
        "global_step": global_step,
        "validation_loss": final_val_loss,
        "validation_perplexity": math.exp(min(final_val_loss, 20)),
        "model_name": args.model_name,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs,
    }
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training_summary.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
