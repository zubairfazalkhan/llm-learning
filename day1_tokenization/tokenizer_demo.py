from transformers import GPT2Tokenizer, GPT2Model
import torch

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")