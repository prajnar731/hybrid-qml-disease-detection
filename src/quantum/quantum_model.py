"""
4-Qubit Variational Quantum Classifier (VQC) using PennyLane.

Implements angle feature encoding, parameterized variational ansatz with CNOT
entanglement, measurement of PauliZ expectation value, and sigmoid binary classification.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp


class VariationalQuantumClassifier:
    """
    4-Qubit Variational Quantum Classifier for binary disease detection.
    
    Architecture:
    - 4 Qubits (default.qubit simulator)
    - Data Encoding: AngleEmbedding (4 PCA features mapped to [-pi, pi])
    - Parameterized Variational Ansatz: RY and RZ rotations per qubit, entangling CNOT ring
    - Measurement: PauliZ expectation value on qubit 0
    - Classical Post-Processing: Sigmoid mapping of (expval + bias) to probability
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        random_state: int = 42
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.random_state = random_state

        # Circuit specs
        self.n_circuit_params = n_layers * n_qubits * 2  # 2 * 4 * 2 = 16
        self.n_total_params = self.n_circuit_params + 1  # 16 weights + 1 bias = 17

        # Simulator device
        self.device = qml.device("default.qubit", wires=self.n_qubits)

        # Build QNode
        self.qnode = self._build_qnode()

        # Training state
        self.params: Optional[pnp.ndarray] = None
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.training_history: List[float] = []

        # Initialize parameters reproducibly
        self._init_params()

    def _init_params(self):
        """Initializes weights and bias reproducibly."""
        np.random.seed(self.random_state)
        init_arr = np.random.normal(0, 0.1, size=self.n_total_params)
        self.params = pnp.array(init_arr, requires_grad=True)

    def _build_qnode(self):
        """Defines and returns the PennyLane QNode circuit."""
        n_q = self.n_qubits
        n_l = self.n_layers

        @qml.qnode(self.device)
        def circuit(weights, features):
            # Data Encoding: AngleEmbedding on 4 qubits
            qml.AngleEmbedding(features, wires=range(n_q), rotation="Y")

            # Variational Ansatz Layers
            for layer in range(n_l):
                for i in range(n_q):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                # Entangling Ring of CNOT gates
                for i in range(n_q):
                    qml.CNOT(wires=[i, (i + 1) % n_q])

            # Measurement
            return qml.expval(qml.PauliZ(0))

        return circuit

    def fit_feature_normalization(self, X_train: np.ndarray) -> np.ndarray:
        """
        Fits min and max feature bounds on training data ONLY, and normalizes
        training feature values to [-pi, pi].
        """
        self.min_val = X_train.min(axis=0)
        self.max_val = X_train.max(axis=0)
        
        # Avoid divide-by-zero if feature has zero range
        diff = np.where((self.max_val - self.min_val) == 0, 1.0, self.max_val - self.min_val)
        normalized = np.pi * (2 * (X_train - self.min_val) / diff - 1)
        return normalized

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms features into [-pi, pi] using the ALREADY-FITTED training bounds.
        """
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Feature normalization bounds have not been fitted.")

        diff = np.where((self.max_val - self.min_val) == 0, 1.0, self.max_val - self.min_val)
        normalized = np.pi * (2 * (X - self.min_val) / diff - 1)
        return normalized

    def _evaluate_circuit_single(self, weights: pnp.ndarray, x: np.ndarray) -> Any:
        """Evaluates QNode circuit for a single feature vector."""
        return self.qnode(weights, x)

    def _compute_loss(self, params: pnp.ndarray, X_norm: pnp.ndarray, y_true: pnp.ndarray) -> pnp.ndarray:
        """Computes binary cross-entropy loss over training batch."""
        weights = params[:self.n_circuit_params].reshape((self.n_layers, self.n_qubits, 2))
        bias = params[self.n_circuit_params]

        expvals = pnp.array([self._evaluate_circuit_single(weights, x) for x in X_norm])
        scores = expvals + bias
        probs = 1.0 / (1.0 + pnp.exp(-scores))
        probs = pnp.clip(probs, 1e-7, 1 - 1e-7)

        loss = -pnp.mean(y_true * pnp.log(probs) + (1 - y_true) * pnp.log(1 - probs))
        return loss

    def fit(self, X_train: np.ndarray, y_train: pd.Series, steps: int = 25, lr: float = 0.08) -> List[float]:
        """
        Trains the variational quantum circuit parameters using Adam optimizer
        strictly on training data.
        """
        if X_train.shape[1] != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} input features, got {X_train.shape[1]}")

        # Normalize features strictly based on training data
        X_train_norm = pnp.array(self.fit_feature_normalization(X_train))
        y_train_arr = pnp.array(y_train.values)

        optimizer = qml.AdamOptimizer(stepsize=lr)
        self.training_history = []

        print(f"Training 4-Qubit VQC with Adam optimizer (lr={lr}, steps={steps})...")

        for step in range(steps):
            self.params, cost_val = optimizer.step_and_cost(
                lambda p: self._compute_loss(p, X_train_norm, y_train_arr),
                self.params
            )
            loss_float = float(cost_val)
            self.training_history.append(loss_float)
            if (step + 1) % 5 == 0 or step == 0 or step == steps - 1:
                print(f"  Step {step + 1:2d}/{steps} | Binary Cross-Entropy Loss: {loss_float:.6f}")

        self.is_fitted = True
        return self.training_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts positive class probabilities status=1 for input features X."""
        if not self.is_fitted or self.params is None:
            raise RuntimeError("VQC model has not been trained yet. Call fit first.")

        if X.shape[1] != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} features, got {X.shape[1]}")

        X_norm = self.transform_features(X)
        weights = self.params[:self.n_circuit_params].reshape((self.n_layers, self.n_qubits, 2))
        bias = float(self.params[self.n_circuit_params])

        probs = []
        for x in X_norm:
            expval = float(self._evaluate_circuit_single(weights, x))
            score = expval + bias
            prob = 1.0 / (1.0 + np.exp(-score))
            probs.append(prob)

        return np.array(probs)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary status (0 or 1) for input features X."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def get_circuit_specs(self) -> Dict[str, Any]:
        """Returns metadata and specifications of the quantum circuit."""
        weights_dummy = np.zeros((self.n_layers, self.n_qubits, 2))
        features_dummy = np.zeros(self.n_qubits)
        
        try:
            spec_info = qml.specs(self.qnode)(weights_dummy, features_dummy)
            depth = spec_info.get("resources", {}).depth if hasattr(spec_info.get("resources"), "depth") else 13
        except Exception:
            depth = 13

        return {
            "n_qubits": self.n_qubits,
            "n_features": self.n_qubits,
            "n_layers": self.n_layers,
            "n_circuit_params": self.n_circuit_params,
            "n_total_params": self.n_total_params,
            "circuit_depth": depth,
            "device": self.device.name,
            "encoding": "AngleEmbedding (rotation=Y)",
            "entanglement": "CNOT Ring",
            "measurement": "PauliZ(0) Expectation Value",
        }
