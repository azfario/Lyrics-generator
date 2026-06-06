from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_INPUT_DIR = Path("dataset lyrics") / "csv"
DEFAULT_OUTPUT_DIR = Path("data") / "processed"
RANDOM_SEED = 42


def fix_mojibake(value: str) -> str:
    """Best-effort repair for common UTF-8 text decoded as Windows-1252."""
    if not isinstance(value, str):
        return ""

    text = value
    if any(marker in text for marker in ("Ã", "Â", "â€")):
        try:
            repaired = text.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
            if repaired.strip():
                text = repaired
        except UnicodeError:
            pass

    replacements = {
        "\u200b": "",
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Â": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_text(value: str) -> str:
    text = fix_mojibake(value).lower()
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\([^\)]*(?:verse|chorus|bridge|intro|outro|hook)[^\)]*\)", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9'\n .,!?-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_lyrics(input_dir: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "Lyric" not in df.columns:
            raise ValueError(f"{csv_file} does not contain a Lyric column")

        keep_columns = [column for column in ("Artist", "Title", "Year", "Lyric") if column in df.columns]
        trimmed = df[keep_columns].copy()
        trimmed["source_file"] = csv_file.name
        frames.append(trimmed)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(
        columns={
            "Artist": "artist",
            "Title": "title",
            "Year": "year",
            "Lyric": "lyric_raw",
        }
    )

    for column in ("artist", "title", "year"):
        if column not in combined.columns:
            combined[column] = ""

    combined["artist"] = combined["artist"].map(fix_mojibake).str.strip()
    combined["title"] = combined["title"].map(fix_mojibake).str.strip()
    combined["lyric"] = combined["lyric_raw"].map(clean_text)
    combined["word_count"] = combined["lyric"].str.split().map(len)
    return combined


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = df["artist"] if df["artist"].value_counts().min() >= 2 else None
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=stratify,
    )

    temp_stratify = temp_df["artist"] if temp_df["artist"].value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=temp_stratify,
    )
    return train_df, val_df, test_df


def add_training_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["lyric"].map(lambda lyric: f"{lyric} <|endoftext|>")
    return df


def write_split(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    selected = df[["artist", "title", "year", "source_file", "word_count", "lyric", "text"]].copy()
    selected.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
    (output_dir / f"{name}.txt").write_text(
        "\n".join(selected["text"].tolist()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare song lyrics data for GPT-2 fine-tuning.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_words", type=int, default=30)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_lyrics(args.input_dir)
    before_rows = len(raw_df)
    cleaned = raw_df.dropna(subset=["lyric"]).copy()
    cleaned = cleaned[cleaned["word_count"] >= args.min_words]
    cleaned = cleaned.drop_duplicates(subset=["lyric"])
    cleaned = cleaned.reset_index(drop=True)
    cleaned = add_training_text(cleaned)

    if cleaned.empty:
        raise ValueError("No lyrics remained after cleaning. Lower --min_words or inspect the CSV files.")

    train_df, val_df, test_df = split_dataset(cleaned)
    write_split(train_df, args.output_dir, "train")
    write_split(val_df, args.output_dir, "validation")
    write_split(test_df, args.output_dir, "test")

    stats = {
        "input_dir": str(args.input_dir),
        "csv_files": len(list(args.input_dir.glob("*.csv"))),
        "raw_rows": int(before_rows),
        "cleaned_rows": int(len(cleaned)),
        "removed_rows": int(before_rows - len(cleaned)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "min_words": int(args.min_words),
        "random_seed": RANDOM_SEED,
        "artist_counts": cleaned["artist"].value_counts().to_dict(),
    }
    (args.output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
