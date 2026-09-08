"""
Preprocessor implementation for UCI Oxford Parkinson's Disease Detection Dataset.

Provides subject-aware train/test splitting, numerical feature scaling,
optional missing value handling, and configurable PCA feature reduction.
"""

from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from .data_loader import extract_subject_ids, EXPECTED_ID_COL, EXPECTED_TARGET_COL


class ParkinsonsPreprocessor:
    """
    Preprocessing pipeline for Parkinson's voice recording dataset.
    
    Ensures zero subject leakage across train/test splits, applies StandardScaler
    fitted exclusively on training data, and provides optional PCA dimensionality reduction.
    """

    def __init__(
        self,
        use_pca: bool = False,
        n_components: Optional[int] = None,
        random_state: int = 42
    ):
        self.use_pca = use_pca
        self.n_components = n_components
        self.random_state = random_state

        # Fitted transformation objects
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None
        self.pca: Optional[PCA] = None
        
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Performs a subject-aware train/test split using GroupShuffleSplit.
        
        Ensures that all recordings from any given subject remain isolated
        in either the training set or the test set.
        """
        rs = random_state if random_state is not None else self.random_state

        # Ensure identifier and target columns exist
        if EXPECTED_ID_COL not in df.columns or EXPECTED_TARGET_COL not in df.columns:
            raise ValueError(
                f"DataFrame must contain '{EXPECTED_ID_COL}' and '{EXPECTED_TARGET_COL}' columns."
            )

        # Extract features (X) excluding identifier and target
        feature_cols = [c for c in df.columns if c not in [EXPECTED_ID_COL, EXPECTED_TARGET_COL]]
        X = df[feature_cols].copy()
        y = df[EXPECTED_TARGET_COL].copy()

        # Extract subject groups
        subjects = extract_subject_ids(df)

        # Group-aware splitting
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_idx, test_idx = next(gss.split(X, y, groups=subjects))

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
        train_subjects = subjects.iloc[train_idx].copy()
        test_subjects = subjects.iloc[test_idx].copy()

        # Mandatory subject isolation check
        train_subject_set = set(train_subjects)
        test_subject_set = set(test_subjects)
        if not train_subject_set.isdisjoint(test_subject_set):
            overlap = train_subject_set.intersection(test_subject_set)
            raise RuntimeError(f"Subject leakage detected! Overlapping subjects: {overlap}")

        return X_train, X_test, y_train, y_test, train_subjects, test_subjects

    def fit_transform_train(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Fits scaler (and optional imputer/PCA) on training data and returns transformed features.
        """
        self.feature_names = list(X_train.columns)
        
        # Check missing values
        if X_train.isnull().sum().sum() > 0:
            self.imputer = SimpleImputer(strategy="median")
            X_train_clean = self.imputer.fit_transform(X_train)
        else:
            self.imputer = None
            X_train_clean = X_train.values if isinstance(X_train, pd.DataFrame) else X_train

        # Fit StandardScaler on training features only
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_clean)

        # Fit PCA if enabled
        if self.use_pca:
            if self.n_components is None:
                raise ValueError("n_components must be specified when use_pca=True")
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            X_train_processed = self.pca.fit_transform(X_train_scaled)
        else:
            self.pca = None
            X_train_processed = X_train_scaled

        self.is_fitted = True
        return X_train_processed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms test or new input data using the ALREADY-FITTED preprocessor.
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit_transform_train first.")

        X_clean = X.values if isinstance(X, pd.DataFrame) else X

        if self.imputer is not None:
            X_clean = self.imputer.transform(X_clean)

        X_scaled = self.scaler.transform(X_clean)

        if self.use_pca and self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        return X_processed

    def get_explained_variance_info(self) -> Optional[Dict[str, Any]]:
        """
        Returns explained variance information if PCA is enabled and fitted.
        """
        if self.use_pca and self.pca is not None:
            ratios = self.pca.explained_variance_ratio_
            return {
                "n_components": self.n_components,
                "explained_variance_ratio": ratios.tolist(),
                "total_explained_variance": float(np.sum(ratios)),
            }
        return None
