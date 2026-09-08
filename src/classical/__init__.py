"""
Classical Machine Learning Baselines for Parkinson's Disease Detection.
"""

from .models import get_classical_models, evaluate_model
from .train_classical import train_and_evaluate_all_models

__all__ = [
    "get_classical_models",
    "evaluate_model",
    "train_and_evaluate_all_models",
]
