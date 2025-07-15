import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Toy Corpus -----
corpus = "i love llms . you love transformers . i hate bugs .".split()
vocab = sorted(set(corpus))
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}
vocab_size = len(vocab)

# ----- Data -----
def make_data(corpus, context_size):
    data = []
    for i in range(len(corpus) - context_size):
        context = corpus[i:i+context_size]
        target = corpus[i+context_size]
        data.append(([word_to_ix[w] for w in context], word_to_ix[target]))
    return data

context_size = 3
data = make_data(corpus, context_size)

# ----- Model -----
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, context_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, context_size, d_model))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=1, batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        tok_embed = self.embedding(x)  # [B, T, D]
        x = tok_embed + self.pos_embedding[:, :x.size(1), :]  # Add pos encoding
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        out = self.fc(attn_out[:, -1, :])  # Predict next token
        return out

# ----- Training Setup -----
model = MiniGPT(vocab_size, d_model=32, context_size=context_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----- Training Loop -----
for epoch in range(300):
    total_loss = 0
    for ctx, target in data:
        x = torch.tensor([ctx])
        y = torch.tensor([target])
        out = model(x)
        loss = loss_fn(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# ----- Generate -----
def generate(seed, n=5):
    model.eval()
    with torch.no_grad():
        context = [word_to_ix[w] for w in seed.split()]
        for _ in range(n):
            x = torch.tensor([context[-context_size:]])
            logits = model(x)
            pred = torch.argmax(logits, dim=-1).item()
            context.append(pred)
        return ' '.join([ix_to_word[i] for i in context])

print("\nGenerated Text:")
print(generate("i love llms", n=5))
