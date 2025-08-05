"""マスクのtop-k予測."""

from transformers.pipelines import pipeline

model_id = "answerdotai/ModernBERT-base"

# fill-maskパイプラインを作成
fill_mask = pipeline("fill-mask", model=model_id, top_k=10)

text = "The movie was full of [MASK]."

# 推論実行
results = fill_mask(text)

for i, output in enumerate(results):
    print(f"Top{i + 1}: token{output['token_str']}| score{output['score']: .4f}")
