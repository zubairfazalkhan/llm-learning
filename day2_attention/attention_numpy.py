import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create toy input vectors
Q = np.array([[1, 0, 1, 0],
              [0, 2, 0, 2],
              [1, 1, 1, 1]], dtype=np.float32)

K = np.array([[1, 0, 1, 0],
              [0, 2, 0, 2],
              [1, 1, 1, 1]], dtype=np.float32)

V = np.array([[0.1, 0.2, 0.3, 0.4],
              [0.5, 0.6, 0.7, 0.8],
              [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)

# Step 2: Compute scaled dot-product attention
dk = Q.shape[-1]
scores = np.matmul(Q, K.T) / np.sqrt(dk)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

weights = softmax(scores)
output = np.matmul(weights, V)

# Step 3: Print results
print("Attention Weights:\n", weights)
print("\nAttention Output:\n", output)

# Step 4: Visualize attention weights
plt.figure(figsize=(6, 5))
sns.heatmap(weights, annot=True, cmap="Blues", xticklabels=["Tok1", "Tok2", "Tok3"], yticklabels=["Tok1", "Tok2", "Tok3"])
plt.title("Attention Weight Heatmap")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.tight_layout()
plt.show()
