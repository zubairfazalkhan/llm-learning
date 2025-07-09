import numpy as np

# Step 1: Create toy input vectors (batch_size=1, seq_len=3, embed_dim=4)
Q = np.array([[1, 0, 1, 0],
              [0, 2, 0, 2],
              [1, 1, 1, 1]], dtype=np.float32)

K = np.array([[1, 0, 1, 0],
              [0, 2, 0, 2],
              [1, 1, 1, 1]], dtype=np.float32)

V = np.array([[0.1, 0.2, 0.3, 0.4],
              [0.5, 0.6, 0.7, 0.8],
              [0.9, 1.0, 1.1, 1.2]], dtype=np.float32)

# Step 2: Compute QK^T
dk = Q.shape[-1]
scores = np.matmul(Q, K.T) / np.sqrt(dk)

# Step 3: Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

weights = softmax(scores)

# Step 4: Attention output
output = np.matmul(weights, V)

print("Attention Weights:\n", weights)
print("\nAttention Output:\n", output)
