# ensure the project root (the folder that contains `src/`) is importable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.clean import get_clean_df
from src.numeralization_embeddings import numerize_embeddings, DEFAULT_MODEL, save_arrays

@pytest.fixture(scope="module")
def clean_df():
    """Load the cleaned dataset."""
    df = get_clean_df()
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns
    return df


def test_embeddings_output_shape_and_type(clean_df, tmp_path):
    """Check that embeddings have the right shape and type."""
    # Run the embedding generation
    X, y, meta = numerize_embeddings(
        df=clean_df,
        text_col="text",
        label_col="label" if "label" in clean_df.columns else None,
        model_name=DEFAULT_MODEL,
        batch_size=8,
        normalize=True,
        device="cpu",
    )

    # --- Basic checks ---
    assert isinstance(X, np.ndarray), "Embeddings should be a numpy array"
    assert X.ndim == 2, "Embeddings must be 2D (samples × dimensions)"
    n_samples, n_dims = X.shape
    assert n_samples == len(clean_df), "Row count must match input data"
    assert np.isfinite(X).all(), "Embeddings must not contain NaNs or infs"
    assert X.dtype == np.float32, "Embeddings should be float32"

    # --- Metadata checks ---
    assert isinstance(meta, dict)
    assert meta["model_name"] == DEFAULT_MODEL

    # --- Save and reload checks ---
    out_prefix = tmp_path / "test_embeddings"
    save_arrays(out_prefix, X, y, meta)
    assert (out_prefix.with_suffix(".npy")).exists(), "Embeddings file not saved"
    if y is not None:
        assert (out_prefix.with_name(out_prefix.stem + "_labels.npy")).exists()
    assert (out_prefix.with_name(out_prefix.stem + "_meta.json")).exists()


def test_embeddings_have_variance(clean_df):
    """Ensure embeddings are not constant (sanity check)."""
    X, _, _ = numerize_embeddings(
        df=clean_df.head(5),  # smaller sample for speed
        text_col="text",
        model_name=DEFAULT_MODEL,
        batch_size=4,
        normalize=True,
        device="cpu",
    )
    stds = X.std(axis=0)
    assert stds.mean() > 0, "Embeddings appear constant — something is wrong"
