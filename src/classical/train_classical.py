"""
Training and evaluation runner for classical machine learning baselines.
"""

from typing import Optional, Dict, Any
import pandas as pd

from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.preprocessor import ParkinsonsPreprocessor
from src.classical.models import get_classical_models, evaluate_model


def train_and_evaluate_all_models(
    use_pca: bool = False,
    n_components: Optional[int] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Loads raw Parkinson's dataset, applies Stage 2 preprocessor (subject-aware split,
    median imputation, StandardScaler, optional PCA), trains all classical ML baseline
    models, and evaluates their performance on unseen test data.
    """
    # 1. Load raw dataset
    df = load_raw_data()

    # 2. Instantiate Stage 2 preprocessor
    preprocessor = ParkinsonsPreprocessor(
        use_pca=use_pca,
        n_components=n_components,
        random_state=random_state
    )

    # 3. Perform subject-aware train/test split
    X_train, X_test, y_train, y_test, train_subjects, test_subjects = preprocessor.split_data(
        df, test_size=0.20, random_state=random_state
    )

    # 4. Preprocessing fitted strictly on training data
    X_train_proc = preprocessor.fit_transform_train(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 5. Fetch classical ML models
    models_dict = get_classical_models(random_state=random_state)

    # 6. Train & Evaluate each model
    evaluations = {}
    table_rows = []

    for name, model in models_dict.items():
        eval_result = evaluate_model(
            model=model,
            X_train=X_train_proc,
            y_train=y_train,
            X_test=X_test_proc,
            y_test=y_test
        )
        evaluations[name] = eval_result

        table_rows.append({
            "Model": name,
            "Accuracy": eval_result["accuracy"],
            "Precision": eval_result["precision"],
            "Recall": eval_result["recall"],
            "F1-Score": eval_result["f1_score"],
            "ROC-AUC": eval_result["roc_auc"],
            "Sensitivity": eval_result["sensitivity"],
        })

    results_df = pd.DataFrame(table_rows)

    return {
        "results_df": results_df,
        "evaluations": evaluations,
        "train_shape": X_train_proc.shape,
        "test_shape": X_test_proc.shape,
        "train_subject_count": len(set(train_subjects)),
        "test_subject_count": len(set(test_subjects)),
        "y_train_dist": y_train.value_counts().to_dict(),
        "y_test_dist": y_test.value_counts().to_dict(),
        "use_pca": use_pca,
        "n_components": n_components,
    }
