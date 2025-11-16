import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.clean import get_clean_df

def test_get_clean_df_returns_dataframe():
    df = get_clean_df()
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns
    assert len(df) > 0

def test_text_column_is_stringy():
    df = get_clean_df()
    sample = df["text"].dropna().head(5).tolist()
    assert all(isinstance(x, str) and x.strip() for x in sample)


