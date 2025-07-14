import numpy as np

# ----- Core Functions -----

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len)))

def masked_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores = scores * mask + (1.0 - mask) * (-1e9)
    weights = softmax(scores)
    return weights @ V

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    return PE

# ----- Vocabulary -----
vocab = ["I", "love", "LLMs", ".", "you", "hate", "<pad>"]
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

# ----- Model Hyperparameters -----
d_model = 8
seq_len = 5

# ----- Embedding (random, fixed) -----
np.random.seed(0)
embedding_matrix = np.random.randn(vocab_size, d_model)

# ----- Weights -----
W_q = np.random.randn(d_model, d_model)
W_k = np.random.randn(d_model, d_model)
W_v = np.random.randn(d_model, d_model)
W_out = np.random.randn(d_model, vocab_size)

# ----- Input Prompt -----
prompt = ["I", "love"]
input_ids = [word_to_idx[w] for w in prompt]

# ----- Autoregressive Generation -----
for _ in range(seq_len - len(prompt)):
    current_seq = input_ids[-seq_len:] if len(input_ids) >= seq_len else input_ids
    padded = [word_to_idx["<pad>"]] * (seq_len - len(current_seq)) + current_seq

    x = embedding_matrix[padded]
    x += positional_encoding(seq_len, d_model)

    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    mask = causal_mask(seq_len)
    attn_out = masked_attention(Q, K, V, mask)

    # Use last token's output to predict next
    last_hidden = attn_out[-1]
    logits = last_hidden @ W_out
    probs = softmax(logits)

    next_token_id = np.argmax(probs)
    input_ids.append(next_token_id)

# ----- Output -----
generated = [idx_to_word[i] for i in input_ids]
print("Generated sequence:", " ".join(generated))
