"""
Numeralization with LLM Embeddings

Turns pre-cleaned text into dense vectors using Sentence-Transformers.
- Default model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Can be used as a standalone script or imported in notebooks/pipelines.

Usage (CLI):
  python -m src.numeralization_embeddings \
      --input data/clean.csv \
      --text-col text \
      --label-col label \
      --out-prefix artifacts/embeddings_minilm

This will create:
  artifacts/embeddings_minilm.npy        (X: float32 [n_samples, 384])
  artifacts/embeddings_minilm_labels.npy (y)
  artifacts/embeddings_minilm_meta.json  (metadata)
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# --- Optional: scikit-learn style transformer for pipelines ---
try:
    from sklearn.base import BaseEstimator, TransformerMixin
except Exception:
    BaseEstimator = object  # type: ignore
    TransformerMixin = object  # type: ignore


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EmbedConfig:
    model_name: str = DEFAULT_MODEL
    batch_size: int = 64
    normalize: bool = True  # cosine-friendly
    device: Optional[str] = None  # "cuda", "mps", or "cpu" (auto if None)


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer.
    Example:
        pipe = Pipeline([
            ("embed", EmbeddingVectorizer()),
            ("clf", LogisticRegression(max_iter=1000))
        ])
    """
    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 batch_size: int = 64,
                 normalize: bool = True,
                 device: Optional[str] = None):
        self.config = EmbedConfig(model_name, batch_size, normalize, device)
        self._model: Optional[SentenceTransformer] = None

    def _ensure_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)

    def fit(self, X: Iterable[str], y=None):  # noqa: N803 (sklearn signature)
        self._ensure_model()
        return self

    def transform(self, X: Iterable[str]):  # noqa: N803 (sklearn signature)
        self._ensure_model()
        texts = list(X)
        vecs = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=True,
        )
        # ensure compact dtype
        return vecs.astype(np.float32)


# --- Convenience, non-sklearn helpers ---

# replace existing load_clean_df with this:

def load_clean_df(input_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load a cleaned dataframe.
    Priority:
      1) explicit CSV path (--input)
      2) try import src.clean.get_clean_df()  (module run from project root)
      3) try import clean.get_clean_df()      (file run from inside src/)
      4) try a common saved CSV path
    """
    if input_path:
        return pd.read_csv(input_path)

    # Try importing the clean module as a package (project-root run)
    try:
        from src import clean as clean_mod  # type: ignore
        for name in ("get_clean_df", "load_clean_df", "clean"):
            if hasattr(clean_mod, name):
                return getattr(clean_mod, name)()
    except Exception:
        pass

    # Try importing the clean module as a sibling (file run from src/)
    try:
        import clean as clean_mod  # type: ignore
        for name in ("get_clean_df", "load_clean_df", "clean"):
            if hasattr(clean_mod, name):
                return getattr(clean_mod, name)()
    except Exception:
        pass

    # Fallback: common cached path produced by your clean step
    fallback = Path("artifacts/preprocessed_news.csv")
    if fallback.exists():
        df = pd.read_csv(fallback)
        # If your text column is 'clean_text', rename for consistency
        if "text" not in df.columns and "clean_text" in df.columns:
            df = df.rename(columns={"clean_text": "text"})
        return df

    raise RuntimeError(
        "No input_path provided and couldn't find a loader in src.clean/clean.\n"
        "Run from the repo root with:  python -m src.numeralization_embeddings\n"
        "Or pass --input artifacts/preprocessed_news.csv"
    )

def numerize_embeddings(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: Optional[str] = "label",
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    normalize: bool = True,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Convert cleaned text column to dense embeddings array X and labels y (if provided).
    Returns (X, y, metadata)
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not in dataframe. Found: {list(df.columns)}")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].values if (label_col and label_col in df.columns) else None

    model = SentenceTransformer(model_name, device=device)
    X = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    ).astype(np.float32)

    meta = {
        "model_name": model_name,
        "embedding_dim": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "normalize": normalize,
        "text_col": text_col,
        "label_col": label_col if label_col in df.columns else None,
    }
    return X, labels, meta


def save_arrays(out_prefix: str | Path, X: np.ndarray, y: Optional[np.ndarray], meta: dict) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_prefix.with_suffix(".npy"), X)
    if y is not None:
        np.save(out_prefix.with_name(out_prefix.stem + "_labels.npy"), y)

    with open(out_prefix.with_name(out_prefix.stem + "_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# --- CLI ---

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Numeralize cleaned text with LLM embeddings.")
    p.add_argument("--input", type=str, default=None,
                   help="CSV path of cleaned data (expects at least a text column). "
                        "If omitted, tries src.clean.get_clean_df().")
    p.add_argument("--text-col", type=str, default="text", help="Name of the text column.")
    p.add_argument("--label-col", type=str, default="label", help="Name of the label column (optional).")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Sentence-Transformers model name.")
    p.add_argument("--batch-size", type=int, default=64, help="Encode batch size.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable L2 normalization of embeddings.")
    p.add_argument("--device", type=str, default=None, help='Force device: "cuda", "mps", or "cpu".')
    p.add_argument("--out-prefix", type=str, default="artifacts/embeddings_minilm",
                   help="Prefix for saved arrays/metadata (without extension).")
    return p.parse_args()


def main():
    args = _parse_args()
    df = load_clean_df(args.input)

    X, y, meta = numerize_embeddings(
        df=df,
        text_col=args.text_col,
        label_col=args.label_col if args.label_col in df.columns else None,
        model_name=args.model,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        device=args.device,
    )

    save_arrays(args.out_prefix, X, y, meta)

    print(f"[ok] embeddings: {X.shape} → saved to:")
    print(f"  {Path(args.out_prefix).with_suffix('.npy')}")
    if y is not None:
        print(f"  {Path(args.out_prefix).with_name(Path(args.out_prefix).stem + '_labels.npy')}")
    print(f"  {Path(args.out_prefix).with_name(Path(args.out_prefix).stem + '_meta.json')}")


if __name__ == "__main__":
    main()
