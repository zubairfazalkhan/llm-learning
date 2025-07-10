import numpy as np

# ----- Utility Functions -----

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def layer_norm(x, epsilon=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + epsilon)

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    return np.matmul(weights, V), weights

# ----- Input -----

# 3 tokens, 4-dim embedding
x = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])

# ----- Step 1: Linear projections (weights simulated randomly) -----
W_q = np.random.randn(4, 4)
W_k = np.random.randn(4, 4)
W_v = np.random.randn(4, 4)

Q = x @ W_q
K = x @ W_k
V = x @ W_v

# ----- Step 2: Self-Attention -----
attn_output, attn_weights = attention(Q, K, V)

# ----- Step 3: Add & Norm -----
x2 = layer_norm(x + attn_output)

# ----- Step 4: Feedforward Network (MLP) -----
W1 = np.random.randn(4, 8)  # Expand
W2 = np.random.randn(8, 4)  # Compress
ffn_output = np.maximum(0, x2 @ W1) @ W2  # ReLU activation

# ----- Step 5: Add & Norm again -----
output = layer_norm(x2 + ffn_output)

# ----- Final Output -----
print("Final Transformer Block Output:\n", output)
