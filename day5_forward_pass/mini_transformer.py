import numpy as np

# ----- Utils -----
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
    return np.matmul(weights, V)

def get_positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    return PE

# ----- Input -----
# Fake embedding for 6 tokens with embedding dim 8
seq_len, d_model = 6, 8
x = np.random.rand(seq_len, d_model)

# ----- Add positional encoding -----
pos_enc = get_positional_encoding(seq_len, d_model)
x += pos_enc

# ----- Transformer Block -----
# Linear projections
W_q, W_k, W_v = np.random.randn(d_model, d_model), np.random.randn(d_model, d_model), np.random.randn(d_model, d_model)
Q, K, V = x @ W_q, x @ W_k, x @ W_v

# Attention
attn_out = attention(Q, K, V)
x2 = layer_norm(x + attn_out)

# Feedforward
W1, W2 = np.random.randn(d_model, d_model*2), np.random.randn(d_model*2, d_model)
ffn = np.maximum(0, x2 @ W1) @ W2
output = layer_norm(x2 + ffn)

# ----- Final Output -----
print("Mini Transformer Output:\n", output)
