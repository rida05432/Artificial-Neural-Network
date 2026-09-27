"""
Plots the training loss curve saved by neural_network.py.
Run neural_network.py first to generate loss_history.csv.
"""

import numpy as np
import matplotlib.pyplot as plt

loss = np.loadtxt("loss_history.csv", delimiter=",", skiprows=1)

plt.figure(figsize=(8, 5))
plt.plot(loss, color="#1F3B57", linewidth=1.5)
plt.title("Training Loss Over Time (Binary Cross-Entropy)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("Saved plot to loss_curve.png")
