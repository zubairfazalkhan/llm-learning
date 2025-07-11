import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / d_model)))
    return PE

# Example parameters
seq_len = 50      # tokens
d_model = 32      # embedding dim

# Generate encoding
pe = get_positional_encoding(seq_len, d_model)

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.title("Positional Encoding Heatmap")
plt.imshow(pe, cmap='viridis', aspect='auto')
plt.xlabel("Embedding Dimensions")
plt.ylabel("Position in Sequence")
plt.colorbar()
plt.tight_layout()
plt.show()
