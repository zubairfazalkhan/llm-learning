import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len)))

def masked_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores = scores * mask + (1.0 - mask) * (-1e9)  # apply mask
    weights = softmax(scores)
    return weights @ V, weights

# ----- Dummy Inputs -----
seq_len, d_model = 5, 8
x = np.random.rand(seq_len, d_model)

# ----- Linear projections -----
W_q, W_k, W_v = np.random.randn(d_model, d_model), np.random.randn(d_model, d_model), np.random.randn(d_model, d_model)
Q, K, V = x @ W_q, x @ W_k, x @ W_v

# ----- Causal mask -----
mask = causal_mask(seq_len)

# ----- Apply masked attention -----
output, attn_weights = masked_attention(Q, K, V, mask)

# ----- Final Output -----
print("Masked Attention Output:\n", output)
print("\nAttention Weights:\n", attn_weights)
