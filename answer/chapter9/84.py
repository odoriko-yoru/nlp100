"""平均による文ベクトル."""

import torch
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import AutoModel
from transformers import AutoTokenizer

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/bert#transformers.BertModel
model = AutoModel.from_pretrained(model_id)

texts = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

cls_last_hidden_state = []

for t in texts:
    # https://huggingface.co/docs/transformers/ja/pad_truncation
    inputs = tokenizer(t, return_tensors="pt", padding=True, truncation=True)

    # 推論
    with torch.no_grad():
        outputs = model(**inputs)

    # 最終層の埋込ベクトルの平均を取得
    # [batch_size, num_of_token, vocabulary_size]
    cls_last_hidden_state.append(torch.mean(outputs.last_hidden_state[0, ...], dim=0))

x = torch.stack(cls_last_hidden_state)

print(pairwise_cosine_similarity(x))
