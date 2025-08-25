"""次単語予測."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# https://github.com/huggingface/huggingface-llama-recipes
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, return_tensors="pt")
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The movie was full of"

with torch.no_grad():
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    prob = torch.softmax(logits[0, -1, :], dim=-1)
    top10_token = torch.topk(prob, 10)

prob_top10 = top10_token.values
idx_top10 = top10_token.indices

word_top10 = tokenizer.decode(idx_top10).split()

print("Predicted word")
for i, (p, w) in enumerate(zip(prob_top10, word_top10, strict=False)):
    print(f"{i + 1} -> {w}\tprob:{p:.4f}")
