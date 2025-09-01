"""プロンプトによる感情分析."""

import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 環境変数の読み込み
DATA_DIR = os.environ.get("DATA_DIR", "")

# Prepare model, tokenizer and templates
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True
)

# Datasetの読み込み
data_dir = Path(DATA_DIR)
dev_df = pd.read_csv(data_dir / "SST-2/dev.tsv", sep="\t")

generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False, adding_side="left"
)

correct = 0

batch_size = 32 if torch.cuda.is_available() else 1

sentences = dev_df["sentence"].tolist()
labels = dev_df["label"].tolist()

for i in tqdm(range(0, len(dev_df), batch_size), total=len(dev_df) / batch_size):
    batch_setntence = sentences[i : i + batch_size]

    batch_chattemplate = []
    batch_labels = []

    for s, label in zip(sentences[i : i + batch_size], labels[i : i + batch_size], strict=False):
        messages = [
            {
                "role": "system",
                "content": """
            You are a classification system for the sentiment analysis.
            If you classificate inputted sentence negative, you should return only 'negative',
            or else (positive sentence) you return only 'positive'.
            For exmaple, the positive sentence 'The movie was full of fun.' is inputted, you shold return 'positive'.
            """,
            },
            {
                "role": "user",
                "content": f"""
            Sentence: {s}
            Emotion:
            """,
            },
        ]

        chat_template = tokenizer.apply_chat_template(messages, tokenize=False, return_tensors="pt")
        batch_chattemplate.append(chat_template)
        batch_labels.append(label)

    # tokenize
    tokenized_batch = tokenizer(
        batch_chattemplate,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    batch_input = tokenized_batch["input_ids"].to(device)
    batch_attention_mask = tokenized_batch["attention_mask"].to(device)
    batch_labels = torch.Tensor(batch_labels).to(device)

    with torch.no_grad():
        # padding_sideが'right'である旨のWarningが出るが
        # batch_inputの中身を見てleft_sideのpaddingになっていることは確認済み
        # attention_maskも正しくマスクしていた
        outputs = model.generate(
            batch_input,
            attention_mask=batch_attention_mask,
            generation_config=generation_config,
            max_new_tokens=32,
        )

    for output, true_label in zip(outputs, batch_labels, strict=False):
        response = tokenizer.decode(output, skip_special_tokens=True).split("assistant")[-1].strip()
        response = response.lower()

        pred_label = 1 if "positive" in response else 0

        if pred_label == true_label:
            correct += 1

accuracy = correct / len(dev_df)

print(f"Accuracy: {accuracy: .4f} ({correct} / {len(dev_df)}))")
