"""
Quantum Machine Learning module for Parkinson's Disease Detection.
"""

from .quantum_model import VariationalQuantumClassifier
from .train_quantum import train_and_evaluate_quantum_model

__all__ = [
    "VariationalQuantumClassifier",
    "train_and_evaluate_quantum_model",
]
