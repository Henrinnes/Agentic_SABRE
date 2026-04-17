#  ←  paste the entire script here

#!/usr/bin/env python3
"""
Dataset Embedding Processor – Colab‑ready
========================================

Updated **23 Apr 2025** to reflect the latest user requirements.

Key updates
-----------
* **Excludes** the columns `label`, `category`, `bucket_start_time`, and
  `time_bucket` when constructing the natural‑language description that gets
  encoded. These columns remain in the final data frame but do **not** influence
  the embedding vectors.
* The generated sentence is always saved as `text_representation`; pass
  `--drop-text` if you explicitly want it removed before writing the Parquet
  file.
* All other behavioural details (streaming, batching, optional `.npy` dumps)
  remain unchanged.

Run in Colab
------------
```bash
!pip install -q sentence-transformers pandas pyarrow tqdm

%%writefile dataset_embedding_processor.py
# ← paste this file (or clone from your repo)

!python dataset_embedding_processor.py /path/to/temp_dataset_features2.csv \
       -o /path/to/processed_chunks
```
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ─── Configuration Defaults ────────────────────────────────────────────────

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast, 384‑dim
DEFAULT_BATCH = 256  # encoder batch size; adjust for GPU RAM
DEFAULT_CHUNK = 250_000  # rows per streaming chunk
TEXT_COL_LIMIT = 256  # max tokens fed to the model

# Columns we do *not* want to embed
EXCLUDE_COLS = {
    "label",
    "category",
    "bucket_start_time",
    "time_bucket",
}

# ─── CLI ────────────────────────────────────────────────────────────────────

import sys

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI *or* notebook cell arguments.

    Colab / Jupyter adds its own flags (e.g. `-f <connection_file>`). We
    therefore use *parse_known_args* and silently ignore anything we don't
    recognise, to prevent `SystemExit: 2` errors when the script is executed in
    a notebook cell with `%run` or similar magic.
    """
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(
        prog="dataset_embedding_processor.py",
        description="Stream a large CSV and add Sentence‑BERT embeddings row‑wise.",
    )
    p.add_argument("csv", type=Path, help="Input CSV file path.")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("./processed_chunks"),
        help="Directory where Parquet chunks will be written.",
    )
    p.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model name or local path compatible with SentenceTransformer.",
    )
    p.add_argument("-b", "--batch", type=int, default=DEFAULT_BATCH, help="Embedding batch size.")
    p.add_argument("-c", "--chunk", type=int, default=DEFAULT_CHUNK, help="CSV chunk size.")
    p.add_argument(
        "--drop-text",
        action="store_true",
        help="Drop the text_representation column before writing parquet.",
    )
    p.add_argument("--to-npy", action="store_true", help="Also dump raw .npy embedding blobs.")

    args, unknown = p.parse_known_args(argv)
    if unknown:
        logging.warning("Ignoring unknown CLI args: %s", unknown)
    return args

# ─── Model Helper ────────────────────────────────────────────────────────── ──────────────────────────────────────────────────────────

def load_model(model_name: str) -> SentenceTransformer:
    logging.info("Loading Sentence‑BERT model %s", model_name)
    model = SentenceTransformer(model_name)
    model.max_seq_length = TEXT_COL_LIMIT  # safety clip
    return model

# ─── Row → Text ────────────────────────────────────────────────────────────

def row_to_text(row: pd.Series) -> str:
    """Turn a dataframe row into a compact natural‑language sentence.

    * Skips columns in **EXCLUDE_COLS**.
    * Omits numeric values equal to 0 (common in this dataset).
    """
    parts: List[str] = []

    pretty_map = {
        "avg_read_throughput": "average read throughput",
        "avg_write_throughput": "average write throughput",
        "read_LBA_variance": "read LBA variance",
        "write_LBA_variance": "write LBA variance",
        "write_entropy_mean": "write entropy mean",
        "mem_read_entropy_mean": "memory read entropy mean",
        "mem_write_entropy_mean": "memory write entropy mean",
        "mem_readwrite_entropy_mean": "memory read‑write entropy mean",
        "mem_read_GPA_variance": "memory read GPA variance",
        "mem_write_GPA_variance": "memory write GPA variance",
        "mem_readwrite_GPA_variance": "memory read‑write GPA variance",
        "mem_exec_GPA_variance": "memory exec GPA variance",
    }

    for col, val in row.items():
        if col in EXCLUDE_COLS or pd.isna(val):
            continue
        if isinstance(val, (int, float)) and val == 0:
            continue  # skip zero‑only metrics for brevity

        label = pretty_map.get(col, col.replace("_", " "))
        parts.append(f"{label} {val}")

    return "; ".join(parts[:50])  # cap to avoid overly long sentences

# ─── Embedding Helper ──────────────────────────────────────────────────────

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        device=model.device,
    ).astype("float32")

# ─── Main Processor ────────────────────────────────────────────────────────

def process_csv(
    csv_file: Path,
    out_dir: Path,
    model: SentenceTransformer,
    batch: int,
    chunk_size: int,
    drop_text: bool = False,
    dump_npy: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, chunk in enumerate(
        tqdm(pd.read_csv(csv_file, chunksize=chunk_size), desc="Reading CSV", unit="chunk")
    ):
        # Create NLP representation
        chunk["text_representation"] = chunk.apply(row_to_text, axis=1)

        # Embed
        embeddings = embed_texts(model, chunk["text_representation"].tolist(), batch)
        chunk["embedding"] = embeddings.tolist()

        # Optionally drop the text column
        if drop_text:
            chunk = chunk.drop(columns="text_representation")

        # Save chunk
        parquet_path = out_dir / f"chunk_{idx:03d}.parquet"
        chunk.to_parquet(parquet_path, index=False)
        logging.info("Saved %s (%d rows)", parquet_path, len(chunk))

        # Optional .npy dump
        if dump_npy:
            npy_path = out_dir / f"chunk_{idx:03d}_embeddings.npy"
            np.save(npy_path, embeddings)
            logging.info("Saved %s", npy_path)

# ─── Entry Point ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model = load_model(args.model)

    process_csv(
        csv_file=args.csv,
        out_dir=args.out,
        model=model,
        batch=args.batch,
        chunk_size=args.chunk,
        drop_text=args.drop_text,
        dump_npy=args.to_npy,
    )


if __name__ == "__main__":
    main()
