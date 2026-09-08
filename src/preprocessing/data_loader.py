"""
Data loading and dataset validation for UCI Oxford Parkinson's Disease Detection Dataset.
"""

import os
import io
import re
import zipfile
import urllib.request
import pandas as pd
import numpy as np

UCI_PARKINSONS_ZIP_URL = "https://archive.ics.uci.edu/static/public/174/parkinsons.zip"
DEFAULT_DATA_PATH = "data/raw/parkinsons.csv"

# Expected dataset attributes according to UCI Dataset ID 174
EXPECTED_TARGET_COL = "status"
EXPECTED_ID_COL = "name"
EXPECTED_NUMERICAL_FEATURE_COUNT = 22


def download_dataset(output_path: str = DEFAULT_DATA_PATH) -> str:
    """
    Downloads the official Oxford Parkinson's Disease Detection Dataset from UCI
    and saves the raw CSV data to output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    print(f"Downloading official UCI Parkinson's dataset from {UCI_PARKINSONS_ZIP_URL}...")
    req = urllib.request.Request(
        UCI_PARKINSONS_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()
            z = zipfile.ZipFile(io.BytesIO(content))
            raw_data = z.read("parkinsons.data")
            with open(output_path, "wb") as f:
                f.write(raw_data)
        print(f"Successfully downloaded raw dataset to {output_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset from official UCI repository ({UCI_PARKINSONS_ZIP_URL}): {e}"
        ) from e

    return output_path


def extract_subject_ids(df: pd.DataFrame) -> pd.Series:
    """
    Extracts subject identity strings from the recording 'name' column.
    
    Format example: 'phon_R01_S01_1' -> Subject ID: 'phon_R01_S01'
    """
    if EXPECTED_ID_COL not in df.columns:
        raise ValueError(f"Column '{EXPECTED_ID_COL}' not found in DataFrame.")

    subject_series = df[EXPECTED_ID_COL].str.extract(r'^(.*)_\d+$')[0]
    if subject_series.isnull().any():
        null_count = subject_series.isnull().sum()
        raise ValueError(
            f"Failed to extract subject identity for {null_count} rows using pattern '^(.*)_\\d+$'."
        )
    
    return subject_series.astype(str)


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validates the dataset structure, types, column presence, target values,
    missing value counts, and feature definitions.
    """
    validation_results = {}

    # Check non-empty
    if df.empty:
        raise ValueError("Validation failed: Dataset is empty.")
    validation_results["row_count"] = len(df)
    validation_results["col_count"] = len(df.columns)

    # Check required columns
    if EXPECTED_TARGET_COL not in df.columns:
        raise ValueError(f"Validation failed: Target column '{EXPECTED_TARGET_COL}' missing.")
    if EXPECTED_ID_COL not in df.columns:
        raise ValueError(f"Validation failed: Identifier column '{EXPECTED_ID_COL}' missing.")

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in [EXPECTED_ID_COL, EXPECTED_TARGET_COL]]
    validation_results["feature_count"] = len(feature_cols)

    if len(feature_cols) != EXPECTED_NUMERICAL_FEATURE_COUNT:
        raise ValueError(
            f"Validation failed: Expected {EXPECTED_NUMERICAL_FEATURE_COUNT} features, got {len(feature_cols)}."
        )

    # Verify target values contain only 0 and 1
    unique_targets = set(df[EXPECTED_TARGET_COL].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Validation failed: Target contains unexpected values {unique_targets}.")
    validation_results["target_distribution"] = df[EXPECTED_TARGET_COL].value_counts().to_dict()

    # Verify feature columns are numeric
    non_numeric_features = [
        c for c in feature_cols if not np.issubdtype(df[c].dtype, np.number)
    ]
    if non_numeric_features:
        raise ValueError(f"Validation failed: Non-numeric feature columns found: {non_numeric_features}")

    # Check missing values
    missing_counts = df.isnull().sum().to_dict()
    total_missing = df.isnull().sum().sum()
    validation_results["missing_value_count"] = total_missing
    validation_results["missing_counts_per_column"] = missing_counts

    # Extract unique subject count
    subjects = extract_subject_ids(df)
    validation_results["unique_subject_count"] = subjects.nunique()

    validation_results["is_valid"] = True
    return validation_results


def load_raw_data(data_path: str = DEFAULT_DATA_PATH, auto_download: bool = True) -> pd.DataFrame:
    """
    Loads raw CSV data, auto-downloading if missing, and validates structure.
    """
    if not os.path.exists(data_path):
        if auto_download:
            download_dataset(data_path)
        else:
            raise FileNotFoundError(f"Raw dataset file not found at {data_path}")

    df = pd.read_csv(data_path)
    validate_dataset(df)
    return df
