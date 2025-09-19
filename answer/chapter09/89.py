"""アーキテクチャの変更.

# Ref: https://github.com/upura/nlp100v2025/blob/update-v2025/ch09/ans89.py
"""

import os
from typing import Dict, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorWithPadding

# 環境変数の読み込み
DATA_DIR = os.environ.get("DATA_DIR", "")


def compute_accuracy(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Calculate accuracy from pred labels and truth labels.

    Parameters
    ----------
    eval_pred : Tuple[np.ndarray, np.ndarray]
        Predicted score

    Returns
    -------
    Dict[str, float]
        Accuracy
    """
    metric = evaluate.load("accuracy")
    pred, labels = eval_pred
    preds = pred.argmax(axis=1)
    return metric.compute(predictions=preds, references=labels)


class SSTDataset(torch.utils.data.Dataset):
    """
    Dataset Class for the SST-2.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 512) -> None:
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        encode = self.tokenizer(self.texts[index], max_length=512, truncation=True, padding=False, return_tensors="pt")

        return {
            "input_ids": encode["input_ids"].squeeze(),
            "attention_mask": encode["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class GoogleBertMaxPoolingClassifier(nn.Module):
    def __init__(self, model_name: str, num_label: int = 2, p_dropout: float = 0.1) -> None:
        super().__init__()
        self.num_label = num_label
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_label)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None):
        # BERTの出力を取得
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 各トークンの最大値プーリング
        # [batch_size, sequence_length, hidden_size] -> [batch_size, hidden_size]
        pooled_output = outputs.last_hidden_state.max(dim=1)[0]

        # ドロップアウトと分類
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # nn.CrossEntropyLoss(logits, labels)が期待する次元は
            # logits : [batch_size, num_classes]
            # labels : [batch_size]
            loss = loss_fct(logits.view(-1, self.num_label), labels.view(-1))

        return {"loss": loss, "logits": logits}


def main() -> None:
    # Datasetの読み込み
    train_tsv = pd.read_csv(f"{DATA_DIR}/SST-2/train.tsv", sep="\t")
    dev_tsv = pd.read_csv(f"{DATA_DIR}/SST-2/dev.tsv", sep="\t")

    train_text = train_tsv["sentence"].tolist()
    dev_text = dev_tsv["sentence"].tolist()

    train_label = train_tsv["label"].tolist()
    dev_label = dev_tsv["label"].tolist()

    # Google BERT
    # https://huggingface.co/google-bert/bert-base-cased
    model_id = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = GoogleBertMaxPoolingClassifier(model_id)

    train_dataset = SSTDataset(train_text, train_label, tokenizer)
    dev_dataset = SSTDataset(dev_text, dev_label, tokenizer)

    # DataCollatorの設定（バッチごとのパディング用）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir="./results_89py",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # fp16=True,
        logging_dir="./logs",
    )

    # DataLoaderも内包されているため
    # collater関数とDataset Classを渡すのみでOK
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_accuracy,
    )

    trainer.train()

    # 最終評価
    eval_results = trainer.evaluate()
    print(f"Accuracy after fine tuning: {eval_results}")


if __name__ == "__main__":
    main()
