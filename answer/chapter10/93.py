"""パープレキシティ(Perplexity: PPL).

Refs:
1. 岡崎直観 他, 自然言語処理の基礎(2023) オーム社 p.133-135
"""

import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Prepare model, tokenizer and texts
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

texts = [
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises",
]

# tokenization
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# config
max_new_tokens = 10
generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Inference
with torch.no_grad():
    outputs = model(**inputs, generation_config=generation_config)
logits = outputs.logits

# PPLの計算
# 言語モデルとi番目の単語の経験分布間のクロスエントロピーを計算
# CrossEntropyLoss = softmax + negative_log-likelihood
cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")


# 文章ごとにクロスエントロピーを算出
for i, text in enumerate(texts):
    # contiguous() の必要性
    # https://openillumi.com/pytorch-memory-efficiency-contiguous-method-guide/
    shift_input_ids = inputs.input_ids[i, 1:].contiguous()
    shift_logits = logits[i, :-1, :].contiguous()

    # 単語ごとのクロスエントロピー
    # 入力token_idが1,それ以外のtoken_idが0の確率分布を渡している
    H_i = cross_entropy_loss(shift_logits, shift_input_ids)

    # 文章のクロスエントロピー
    H = H_i.mean()

    # PPL
    # 底にはネイピア数または2が用いられるが、ここではネイピア数を採用
    ppl = math.e**H

    print(f"""Sentence: {text}\nPPL: {ppl: .4f}\n""")
