"""極性分析."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "./results/checkpoint-6315"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

texts = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]

for t in texts:
    # tokenize
    inputs = tokenizer(t, return_tensors="pt", padding=False, truncation=True)

    # 推論
    with torch.no_grad():
        outputs = model(**inputs)

    # 確率分布を計算（dim=1でクラス間の確率を正規化）
    probs = torch.softmax(outputs.logits, dim=1)

    # 予測ラベルを取得
    pred_label = torch.argmax(outputs.logits, dim=1).item()

    # 結果を表示
    print(f"{t} -> {'negative' if pred_label == 0 else 'positive'} (confidence: {probs.max().item():.3f})")
