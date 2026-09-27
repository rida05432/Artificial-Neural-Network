"""
Artificial Neural Network — implemented from scratch with NumPy.

This is a multilayer perceptron (MLP) built without using any deep-learning
library's built-in model (no sklearn.neural_network, no PyTorch, no
TensorFlow). Every part of the forward pass, backpropagation, and gradient
descent update is written by hand, so you can explain exactly what happens
at each step in an interview.

Dataset: Breast Cancer Wisconsin (Diagnostic) dataset, loaded from
scikit-learn. Binary classification task: predict whether a tumour is
malignant (1) or benign (0) from 30 numeric features (cell measurements).

Architecture: configurable fully-connected network, default
30 (input) -> 16 (hidden, ReLU) -> 8 (hidden, ReLU) -> 1 (output, Sigmoid)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ---------------------------------------------------------------------------
# Activation functions and their derivatives
# ---------------------------------------------------------------------------

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def sigmoid(z):
    # Clip to avoid overflow in exp() for very negative/positive inputs
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


# ---------------------------------------------------------------------------
# The neural network itself
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    A multilayer perceptron with an arbitrary number of hidden layers.

    layer_sizes: list of ints, e.g. [30, 16, 8, 1]
        First entry = number of input features.
        Last entry = number of output units (1 for binary classification).
        Everything in between = hidden layer sizes.
    """

    def __init__(self, layer_sizes, learning_rate=0.05, seed=42):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes) - 1

        rng = np.random.default_rng(seed)

        # He initialisation for weights feeding into ReLU layers, which
        # keeps the variance of activations stable as signals pass through
        # deeper layers (naive small-random-weight init tends to vanish).
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(2.0 / fan_in)
            W = rng.normal(0, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

        self.loss_history = []

    def forward(self, X):
        """
        Runs the forward pass and caches every intermediate value needed
        for backpropagation (the pre-activation 'z' and post-activation 'a'
        at every layer).
        """
        activations = [X]
        zs = []

        a = X
        for i in range(self.num_layers):
            z = a @ self.weights[i] + self.biases[i]
            zs.append(z)

            is_output_layer = (i == self.num_layers - 1)
            a = sigmoid(z) if is_output_layer else relu(z)
            activations.append(a)

        return activations, zs

    def backward(self, activations, zs, y_true):
        """
        Backpropagation: computes the gradient of the binary cross-entropy
        loss with respect to every weight and bias, layer by layer, working
        backwards from the output.
        """
        m = y_true.shape[0]
        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        y_pred = activations[-1]

        # dL/dz for the output layer. For sigmoid output + binary
        # cross-entropy loss, this simplifies neatly to (y_pred - y_true).
        delta = y_pred - y_true

        for layer in reversed(range(self.num_layers)):
            a_prev = activations[layer]

            grads_w[layer] = (a_prev.T @ delta) / m
            grads_b[layer] = np.sum(delta, axis=0, keepdims=True) / m

            if layer > 0:
                # Propagate the error signal back through this layer's
                # weights, then through the ReLU derivative of the layer
                # before it, ready for the next iteration.
                delta = (delta @ self.weights[layer].T) * relu_derivative(zs[layer - 1])

        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b):
        """Vanilla gradient descent step: move each parameter a small
        distance opposite to its gradient."""
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    @staticmethod
    def binary_cross_entropy(y_true, y_pred, eps=1e-9):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def train(self, X, y, epochs=500, verbose_every=50):
        for epoch in range(1, epochs + 1):
            activations, zs = self.forward(X)
            loss = self.binary_cross_entropy(y, activations[-1])
            self.loss_history.append(loss)

            grads_w, grads_b = self.backward(activations, zs, y)
            self.update_parameters(grads_w, grads_b)

            if verbose_every and epoch % verbose_every == 0:
                acc = accuracy_score(y, (activations[-1] > 0.5).astype(int))
                print(f"Epoch {epoch:4d} | loss = {loss:.4f} | train accuracy = {acc:.4f}")

    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_data(test_size=0.2, seed=42):
    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Standardise features (zero mean, unit variance). Fit ONLY on the
    # training set, then apply the same transform to the test set, to avoid
    # leaking test-set statistics into training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, data.feature_names, data.target_names


# ---------------------------------------------------------------------------
# Main: train, evaluate, and report results
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Artificial Neural Network from scratch — Breast Cancer Diagnosis")
    print("=" * 70)

    X_train, X_test, y_train, y_test, feature_names, target_names = load_data()
    print(f"\nDataset: {X_train.shape[0]} training examples, {X_test.shape[0]} test examples")
    print(f"Features: {X_train.shape[1]} (e.g. {', '.join(feature_names[:3])}, ...)")
    print(f"Classes: {list(target_names)} (0 = malignant, 1 = benign)\n")

    # Architecture: 30 inputs -> 16 -> 8 -> 1 output
    nn = NeuralNetwork(layer_sizes=[X_train.shape[1], 16, 8, 1], learning_rate=0.05)

    print("Training...\n")
    nn.train(X_train, y_train, epochs=500, verbose_every=50)

    print("\n" + "-" * 70)
    print("Final evaluation on held-out test set")
    print("-" * 70)

    y_pred = nn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"\nConfusion matrix:\n{cm}")
    print("(rows = actual, columns = predicted, order = [malignant, benign])")

    # Save a simple loss curve so training behaviour can be inspected/plotted
    np.savetxt("loss_history.csv", nn.loss_history, delimiter=",", header="loss", comments="")
    print("\nSaved training loss history to loss_history.csv")


if __name__ == "__main__":
    main()
