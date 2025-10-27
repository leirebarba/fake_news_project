import csv
import sys
import pickle
import subprocess
from pathlib import Path

import numpy as np
from scipy import sparse


def _run_vectorize(tmpdir: Path, rows, *, min_df=1, ngram=(1, 1)):
    """Create a tiny CSV, run src.vectorize, return paths to outputs."""
    # 1) write a minimal cleaned CSV with a 'clean_text' column
    csv_path = tmpdir / "mini.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clean_text"])
        for r in rows:
            w.writerow([r])

    # 2) choose output paths
    out_vec = tmpdir / "tfidf_vectorizer.pkl"
    out_npz = tmpdir / "X_tfidf.npz"
    out_vocab = tmpdir / "tfidf_features.txt"

    # 3) run: python -m src.vectorize ...
    cmd = [
        sys.executable, "-m", "src.vectorize",
        "--input", str(csv_path),
        "--text-col", "clean_text",
        "--out-vectorizer", str(out_vec),
        "--out-features", str(out_npz),
        "--vocab-txt", str(out_vocab),
        "--min-df", str(min_df),
        "--ngram-min", str(ngram[0]),
        "--ngram-max", str(ngram[1]),
        "--max-features", "0",  # unlimited
    ]
    subprocess.check_call(cmd)
    return out_vec, out_npz, out_vocab


def test_builds_matrix_and_files(tmp_path: Path):
    rows = [
        "new york city is big",
        "new york state is large",
        "paris is beautiful",
    ]
    out_vec, out_npz, out_vocab = _run_vectorize(tmp_path, rows, min_df=1)

    assert out_vec.exists()
    assert out_npz.exists()
    assert out_vocab.exists()

    X = sparse.load_npz(out_npz)
    assert X.shape[0] == 3               # 3 docs
    assert X.shape[1] > 0                # some features created

    vocab = set(Path(out_vocab).read_text(encoding="utf-8").splitlines())
    # stopwords like "is" should be removed by the English stopword list
    assert "is" not in vocab
    # a meaningful token should exist
    assert "york" in vocab or "paris" in vocab


def test_vectorizer_pickle_roundtrip(tmp_path: Path):
    rows = [
        "cat sat on the mat",
        "dog sat on the rug",
    ]
    out_vec, out_npz, out_vocab = _run_vectorize(tmp_path, rows, min_df=1)

    # load saved artifacts
    X_saved = sparse.load_npz(out_npz).tocsr()
    with open(out_vec, "rb") as f:
        vec = pickle.load(f)

    # re-transform the same docs using the saved vectorizer;
    # result should be same shape and numerically equal
    X_again = vec.transform(rows)
    assert X_again.shape == X_saved.shape
    np.testing.assert_allclose(X_again.toarray(), X_saved.toarray(), rtol=1e-6, atol=1e-8)


def test_ngram_range_includes_bigrams(tmp_path: Path):
    rows = [
        "new york is great",
        "i love new york",
    ]
    _, _, out_vocab = _run_vectorize(tmp_path, rows, min_df=1, ngram=(1, 2))

    vocab = set(Path(out_vocab).read_text(encoding="utf-8").splitlines())
    # Expect the bigram to show up when ngram_max=2
    assert "new york" in vocab


def test_min_df_filters_rare_terms(tmp_path: Path):
    rows = [
        "alpha beta gamma",   # alpha appears 2x overall
        "alpha delta",        # beta, gamma, delta appear once each
    ]
    # min_df=1 => keep everything (except stopwords)
    _, out_npz1, _ = _run_vectorize(tmp_path / "m1", rows, min_df=1)
    # min_df=2 => only terms that appear in BOTH docs survive (alpha)
    _, out_npz2, _ = _run_vectorize(tmp_path / "m2", rows, min_df=2)

    X1 = sparse.load_npz(out_npz1)
    X2 = sparse.load_npz(out_npz2)

    assert X1.shape[1] > X2.shape[1]     # fewer features with higher min_df
    # with min_df=2 and these rows, expect only 'alpha' to remain
    assert X2.shape[1] == 1