"""
RDset semantic-embedding pipeline: the single definition of how RDset PE
metadata becomes the Semantic Agent's 384-D input.

Identifier columns are dropped immediately after loading, before any other
processing. `Name` and `md5` identify a file rather than describe it, and in
RDset `Name` also encodes the label: every ransomware row is named
`VirusShare_<hash>` and no benign row is, so embedding it lets the model
classify by file name alone (AUROC 1.0 from that field by itself). This
matches the IEEE Access preprocessing ("variables 'Md5' and 'Name' are
dropped to avoid confounding model development").

Usage:
    python data_pipeline/rdset_semantic.py Ransomware.csv semantic_dataset_with_embeddings.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 256

# Dropped before anything else touches the data.
IDENTIFIER_COLUMNS = ["Name", "md5"]

# Raw header fields and memory-layout constants with no semantic signal.
LOW_SIGNAL_COLUMNS = [
    "SizeOfOptionalHeader", "MajorLinkerVersion", "MinorLinkerVersion",
    "BaseOfCode", "BaseOfData", "ImageBase",
    "SectionAlignment", "FileAlignment", "CheckSum",
    "SizeOfHeapReserve", "SizeOfHeapCommit",
    "SizeOfStackReserve", "SizeOfStackCommit",
    "LoaderFlags", "NumberOfRvaAndSizes",
]

# PE metadata fields joined into the text that is embedded.
SEMANTIC_COLUMNS = [
    "Machine", "Characteristics",
    "ImportsNbDLL", "ImportsNb", "ExportNb",
    "SectionsMeanEntropy", "SectionsMaxEntropy", "SectionsMinEntropy",
    "ResourcesNb", "ResourcesMeanEntropy", "ResourcesMinEntropy",
    "ResourcesMaxEntropy", "ResourcesMeanSize", "ResourcesMinSize",
    "ResourcesMaxSize", "LoadConfigurationSize", "VersionInformationSize",
    "Subsystem", "DllCharacteristics",
]

assert not set(IDENTIFIER_COLUMNS) & set(SEMANTIC_COLUMNS)


def load_rdset(csv_path: str | Path) -> pd.DataFrame:
    """Load Ransomware.csv with identifiers removed and `label` (1 = ransomware) added."""
    df = pd.read_csv(csv_path, delimiter="|", engine="python")
    df = df.drop(columns=IDENTIFIER_COLUMNS)
    df["label"] = (df["legitimate"] == 0).astype(int)
    return df.drop(columns=["legitimate"] + LOW_SIGNAL_COLUMNS, errors="ignore")


def semantic_text(df: pd.DataFrame) -> pd.Series:
    """Join the semantic columns present in `df` into one string per row."""
    cols: List[str] = [c for c in SEMANTIC_COLUMNS if c in df.columns]
    return df[cols].astype(str).agg(" ".join, axis=1)


def load_encoder(device: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


def build_embeddings(csv_path: str | Path, out_path: str | Path, device: str | None = None) -> pd.DataFrame:
    """Write the identifier-free RDset columns plus `semantic_text` and the 384-D `embedding`."""
    df = load_rdset(csv_path)
    df["semantic_text"] = semantic_text(df)
    emb = load_encoder(device).encode(df["semantic_text"].tolist(), batch_size=BATCH_SIZE,
                                      show_progress_bar=True, convert_to_numpy=True)
    df["embedding"] = emb.astype(np.float32).tolist()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv", help="RDset Ransomware.csv (pipe-delimited)")
    p.add_argument("out", help="output parquet")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    out = build_embeddings(args.csv, args.out, args.device)
    print(f"Wrote {len(out):,} rows ({EMBEDDING_MODEL}, identifiers dropped) -> {args.out}")


if __name__ == "__main__":
    main()
