"""
Step 1 — Clean raw news text using repo-relative paths (no vectorization, no models).

Behavior:
- Detect repo root by finding a parent folder with a `.git` directory.
- Use <repo>/data/sample_news_dataset.csv if present; otherwise the first .csv in <repo>/data.
- Save to <repo>/artifacts/preprocessed_news.csv.
- Add a 'clean_text' column: lowercase, remove URLs, punctuation, numbers, extra spaces.

Optional overrides:
    python scripts/clean_text.py --input data/other.csv --out-csv artifacts/out.csv --text-col text
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd


# ---------- repo utilities ----------
def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return start.resolve().parent  # fallback


def pick_input_csv(repo_root: Path) -> Path:
    data_dir = repo_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected a 'data' folder at: {data_dir}")
    preferred = data_dir / "sample_news_dataset.csv"
    if preferred.exists():
        return preferred
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in: {data_dir}")
    return csvs[0]


# ---------- cleaning ----------
def basic_clean(s: str) -> str:
    """Lowercase, remove URLs, non-letters, collapse spaces."""
    if not isinstance(s, str):
        s = "" if s is np.nan else str(s)
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)     # remove URLs
    s = re.sub(r"[^a-z\s]", " ", s)             # keep only letters/spaces
    s = re.sub(r"\s+", " ", s).strip()          # collapse spaces
    return s


POSSIBLE_TEXT_COLS = [
    "text", "content", "article", "body", "full_text", "news", "message",
    "Text", "Message"
]


def infer_text_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower() in [p.lower() for p in POSSIBLE_TEXT_COLS]:
            return c
    # fallback: prefer title/text pair, else longest avg string column
    title_like, text_like = None, None
    for c in df.columns:
        if c.lower() in ["title", "headline", "subject"]:
            title_like = c
        if c.lower() in ["text", "content", "article", "body", "full_text", "news", "message"]:
            text_like = c
    if text_like:
        return text_like
    if title_like:
        return title_like
    string_cols = [c for c in df.columns if df[c].dtype == object]
    if string_cols:
        avg_len = {c: df[c].astype(str).str.len().mean() for c in string_cols}
        return max(avg_len, key=avg_len.get)
    raise ValueError("Could not infer a text column automatically. Pass --text-col.")


def main():
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Optional CLI overrides
    ap = argparse.ArgumentParser(
        description="Clean raw text into a 'clean_text' column (repo-relative defaults).",
        add_help=True
    )
    ap.add_argument("--input", type=str, default=None,
                    help="(Optional) Input CSV path, relative to repo root or absolute.")
    ap.add_argument("--out-csv", type=str, default=None,
                    help="(Optional) Output CSV path, relative to repo root or absolute.")
    ap.add_argument("--text-col", type=str, default=None,
                    help="(Optional) Name of the text column.")
    args = ap.parse_args()

    # Resolve input/output
    in_path = Path(args.input).resolve() if args.input else pick_input_csv(repo_root)
    if not in_path.is_absolute():
        in_path = (repo_root / in_path).resolve()

    out_path = Path(args.out_csv).resolve() if args.out_csv else (artifacts_dir / "preprocessed_news.csv")
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load CSV (be forgiving with encodings)
    try:
        df = pd.read_csv(in_path)
    except UnicodeDecodeError:
        df = pd.read_csv(in_path, encoding="latin-1")

    text_col = args.text_col or infer_text_column(df)

    # Clean
    df["clean_text"] = df[text_col].astype(str).map(basic_clean)

    # Save
    df.to_csv(out_path, index=False)

    # Console info
    rel = lambda p: str(p.relative_to(repo_root)) if repo_root in p.parents else str(p)
    print("=== Cleaning complete ===")
    print(f"Repo root: {repo_root}")
    print(f"Input CSV: {rel(in_path)}")
    print(f"Text col : {text_col}")
    print(f"Saved to : {rel(out_path)}")
    print("Preview:")
    print(df[[text_col, "clean_text"]].head(5))


if __name__ == "__main__":
    main()

    # --- add this at the end of src/clean.py ---

def get_clean_df() -> pd.DataFrame:
    """Return the cleaned dataset so other scripts (like embeddings) can import it."""
    path = Path("artifacts/preprocessed_news.csv")
    if not path.exists():
        raise FileNotFoundError(
            "Run src/clean.py first to generate artifacts/preprocessed_news.csv"
        )

    df = pd.read_csv(path)

    # Keep only the cleaned text column and rename it to 'text'
    if "clean_text" in df.columns:
        # drop any duplicate 'text' columns that might already exist
        dup_text_cols = [c for c in df.columns if c == "text"]
        if dup_text_cols:
            df = df.drop(columns=dup_text_cols)
        df = df.rename(columns={"clean_text": "text"})

    return df

