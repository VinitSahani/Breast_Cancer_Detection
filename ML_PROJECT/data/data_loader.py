"""
=============================================================================
AGENT 1: DATA & PREPROCESSING AGENT
=============================================================================
Purpose : Load, validate, and preprocess the Breast Cancer Wisconsin dataset.
Input   : None (loads from sklearn.datasets)
Output  : A structured dictionary with train/test splits, scaler, and metadata
          so that downstream agents can consume it without re-processing.

Author  : Data Processing Expert (Agent 1)
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# PUBLIC API — the only function other agents need to call
# ---------------------------------------------------------------------------

def load_and_preprocess(random_state: int = 42, test_size: float = 0.20) -> dict:
    """
    End-to-end data pipeline: load → validate → scale → split.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility across all agents.
    test_size : float
        Fraction of the dataset reserved for testing (default 20 %).

    Returns
    -------
    dict
        {
            "X_train"       : np.ndarray   — scaled training features,
            "X_test"        : np.ndarray   — scaled testing features,
            "y_train"       : pd.Series    — training labels,
            "y_test"        : pd.Series    — testing labels,
            "scaler"        : StandardScaler (fitted),
            "feature_names" : list[str]    — original feature column names,
            "raw_df"        : pd.DataFrame — unscaled data with target column
        }
    """

    # ------------------------------------------------------------------
    # STEP 1  —  Load the raw dataset from sklearn
    # ------------------------------------------------------------------
    # The Breast Cancer Wisconsin dataset contains 569 samples with
    # 30 numeric features computed from digitised images of fine-needle
    # aspirate (FNA) of a breast mass.
    # ------------------------------------------------------------------
    data = load_breast_cancer()

    # ------------------------------------------------------------------
    # STEP 2  —  Build a clean DataFrame with readable column names
    # ------------------------------------------------------------------
    # Using the feature names shipped with the dataset keeps things
    # consistent and self-documenting.
    # ------------------------------------------------------------------
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Map numeric target to human-readable labels:
    #   0 → Malignant  (cancerous — the dangerous class)
    #   1 → Benign     (non-cancerous)
    label_map = {0: "Malignant", 1: "Benign"}
    df["target"] = pd.Series(data.target).map(label_map)

    print("=" * 65)
    print("  AGENT 1 — DATA & PREPROCESSING")
    print("=" * 65)

    # ------------------------------------------------------------------
    # STEP 3  —  Data validation
    # ------------------------------------------------------------------
    # Before any modelling we must confirm the data is clean:
    #   • No missing values (NaNs would break most ML algorithms)
    #   • All feature columns are numeric (required for scaling)
    #   • Shape & head give a quick sanity check
    # ------------------------------------------------------------------
    print("\n📊  Dataset Shape :", df.shape)
    print("\n📋  First 5 rows:")
    print(df.head().to_string())

    missing = df.isnull().sum().sum()
    print(f"\n🔍  Missing values : {missing}")
    if missing > 0:
        print("   ⚠  WARNING — missing values found! Consider imputation.")
    else:
        print("   ✅  No missing values — dataset is clean.")

    print(f"\n🔢  Data types:\n{df.dtypes.value_counts().to_string()}")
    print(f"\n📈  Dataset info:")
    print(f"   Samples  : {df.shape[0]}")
    print(f"   Features : {df.shape[1] - 1}")  # exclude target column
    print(f"   Target classes : {df['target'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # STEP 4  —  Feature scaling with StandardScaler
    # ------------------------------------------------------------------
    # WHY?  Many algorithms (SVM, KNN, Logistic Regression with
    #        regularisation) are sensitive to feature magnitudes.
    #        StandardScaler transforms each feature to zero mean and
    #        unit variance, ensuring no single feature dominates.
    #
    # IMPORTANT: We fit the scaler on the TRAINING data only to avoid
    #            data leakage from the test set.
    # ------------------------------------------------------------------
    feature_cols = data.feature_names.tolist()
    X = df[feature_cols].values
    y = df["target"]

    # ------------------------------------------------------------------
    # STEP 5  —  Train / Test split (80 / 20)
    # ------------------------------------------------------------------
    # WHY?  Holding out 20 % of the data lets Agent 3 evaluate model
    #        performance on unseen samples, giving an honest estimate
    #        of generalisation ability.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # preserve class balance in both splits
    )

    # Fit scaler on training set, then transform both sets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\n✂️  Train / Test split (test_size={test_size}, seed={random_state}):")
    print(f"   X_train : {X_train_scaled.shape}")
    print(f"   X_test  : {X_test_scaled.shape}")
    print(f"   y_train : {y_train.shape}  —  {y_train.value_counts().to_dict()}")
    print(f"   y_test  : {y_test.shape}  —  {y_test.value_counts().to_dict()}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # STEP 6  —  Package outputs for downstream agents
    # ------------------------------------------------------------------
    output = {
        "X_train"       : X_train_scaled,
        "X_test"        : X_test_scaled,
        "y_train"       : y_train.reset_index(drop=True),
        "y_test"        : y_test.reset_index(drop=True),
        "scaler"        : scaler,
        "feature_names" : feature_cols,
        "raw_df"        : df,
    }

    return output


# ---------------------------------------------------------------------------
# Stand-alone execution — useful for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = load_and_preprocess()
    print("\n✅  Agent 1 completed.  Keys returned:")
    for k, v in result.items():
        desc = type(v).__name__
        if hasattr(v, "shape"):
            desc += f"  shape={v.shape}"
        print(f"   • {k:16s} → {desc}")
