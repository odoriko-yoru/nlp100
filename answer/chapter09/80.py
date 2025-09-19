"""トークン化.

Refs:
1. https://huggingface.co/answerdotai/ModernBERT-base#usage
"""

from transformers import AutoTokenizer

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

text = "The movie was full of incomprehensibilities."

inputs = tokenizer(text, return_tensors="pt")

print(inputs)
