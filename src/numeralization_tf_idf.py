"""
Step 2 — Turn cleaned text into numbers (TF-IDF), repo-relative.

Default behavior:
- Input:  <repo>/artifacts/preprocessed_news.csv
          (fallback: <repo>/data/sample_news_dataset.csv or first CSV under data/)
- Output: <repo>/artifacts/tfidf_vectorizer.pkl
          <repo>/artifacts/X_tfidf.npz
          <repo>/artifacts/tfidf_features.txt (vocab)

CLI overrides (optional):
    python -m src.vectorize \
        --input artifacts/preprocessed_news.csv \
        --out-vectorizer artifacts/tfidf_vectorizer.pkl \
        --out-features artifacts/X_tfidf.npz \
        --vocab-txt artifacts/tfidf_features.txt \
        --text-col clean_text \
        --min-df 2 --ngram-min 1 --ngram-max 1 --max-features 0
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------- small utilities (mirror behavior from clean.py) ----------

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return start.resolve().parent  # fallback

def pick_clean_csv(repo_root: Path) -> Path:
    """Prefer artifacts/preprocessed_news.csv, else look in data/ for a CSV."""
    art = repo_root / "artifacts" / "preprocessed_news.csv"
    if art.exists():
        return art
    data_dir = repo_root / "data"
    preferred = data_dir / "sample_news_dataset.csv"
    if preferred.exists():
        return preferred
    if data_dir.exists():
        csvs = sorted(data_dir.glob("*.csv"))
        if csvs:
            return csvs[0]
    raise FileNotFoundError(
        "No input CSV found. Run the cleaning step first (src.clean) or pass --input."
    )

# Minimal cleaner only used if 'clean_text' is missing
def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is np.nan else str(s)
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(
        description="Vectorize cleaned text to TF-IDF (repo-relative defaults)."
    )
    ap.add_argument("--input", type=str, default=None,
                    help="Input CSV (default: artifacts/preprocessed_news.csv or a CSV under data/).")
    ap.add_argument("--text-col", type=str, default=None,
                    help="Name of the text column (default: 'clean_text' if present, else tries to infer).")

    ap.add_argument("--out-vectorizer", type=str, default=None,
                    help="Path to save fitted TfidfVectorizer.pkl (default: artifacts/tfidf_vectorizer.pkl).")
    ap.add_argument("--out-features", type=str, default=None,
                    help="Path to save TF-IDF sparse matrix .npz (default: artifacts/X_tfidf.npz).")
    ap.add_argument("--vocab-txt", type=str, default=None,
                    help="Optional path to save feature names (default: artifacts/tfidf_features.txt).")

    # TF-IDF knobs
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--max-features", type=int, default=0, help="0 means unlimited.")
    ap.add_argument("--ngram-min", type=int, default=1)
    ap.add_argument("--ngram-max", type=int, default=1)

    args = ap.parse_args()

    # Resolve input
    in_path = Path(args.input).resolve() if args.input else pick_clean_csv(repo_root)
    if not in_path.is_absolute():
        in_path = (repo_root / in_path).resolve()

    # Resolve outputs
    out_vec = Path(args.out_vectorizer).resolve() if args.out_vectorizer else (artifacts_dir / "tfidf_vectorizer.pkl")
    out_npz = Path(args.out_features).resolve()   if args.out_features   else (artifacts_dir / "X_tfidf.npz")
    out_vocab = Path(args.vocab_txt).resolve()    if args.vocab_txt      else (artifacts_dir / "tfidf_features.txt")
    out_vec.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_vocab.parent.mkdir(parents=True, exist_ok=True)

    # Load CSV
    try:
        df = pd.read_csv(in_path)
    except UnicodeDecodeError:
        df = pd.read_csv(in_path, encoding="latin-1")

    # Choose the text column
    text_col = args.text_col
    if text_col is None:
        if "clean_text" in df.columns:
            text_col = "clean_text"
        else:
            # Try common names; else fall back to the longest string column
            candidates = ["text", "content", "article", "body", "full_text", "news", "message"]
            for c in df.columns:
                if c.lower() in candidates:
                    text_col = c
                    break
            if text_col is None:
                string_cols = [c for c in df.columns if df[c].dtype == object]
                if string_cols:
                    avg_len = {c: df[c].astype(str).str.len().mean() for c in string_cols}
                    text_col = max(avg_len, key=avg_len.get)
                else:
                    raise ValueError("Could not infer a text column; pass --text-col.")

    # Ensure we have clean text
    series = df[text_col].astype(str)
    if text_col != "clean_text":
        # If the column wasn't clean_text, lightly clean it just in case
        series = series.map(basic_clean)

    # Vectorize
    max_feats = None if args.max_features in (None, 0) else int(args.max_features)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=int(args.min_df),
        max_features=max_feats,
        ngram_range=(int(args.ngram_min), int(args.ngram_max)),
    )
    X = vectorizer.fit_transform(series.values)

    # Save artifacts
    with open(out_vec, "wb") as f:
        pickle.dump(vectorizer, f)
    sparse.save_npz(out_npz, X)
    Path(out_vocab).write_text("\n".join(vectorizer.get_feature_names_out()), encoding="utf-8")

    # Console info
    rel = lambda p: str(p.relative_to(repo_root)) if repo_root in p.parents else str(p)
    print("=== TF-IDF vectorization complete ===")
    print(f"Input CSV    : {rel(in_path)} [col: {text_col}]")
    print(f"Docs x Feats : {X.shape[0]} x {X.shape[1]}")
    print(f"Saved vector : {rel(out_vec)}")
    print(f"Saved matrix : {rel(out_npz)}")
    print(f"Saved vocab  : {rel(out_vocab)}")


if __name__ == "__main__":
    main()
