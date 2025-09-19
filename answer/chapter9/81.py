"""マスクの予測.

Refs:
1. https://huggingface.co/answerdotai/ModernBERT-base#usage
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

text = "The movie was full of [MASK]."

inputs = tokenizer(text, return_tensors="pt")

# 推論
outputs = model(**inputs)

# output logits size
print(outputs.logits.size())

# [MASK]のindexを取得
masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)

# [MASK]のlogitの値が最大の辞書indexを取得
predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)

# logitが最大であったtokenを取得
predicted_token = tokenizer.decode(predicted_token_id)
print("Predicted token:", predicted_token)
