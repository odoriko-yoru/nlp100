"""ファインチューニング.

コードは以下の書籍とGitHubを参考にした。

山田育矢、鈴木正敏、山田康輔、李凌寒 (2023) 「大規模言語モデル入門」技術評論社
https://github.com/ghmagazine/llm-book/tree/main/chapter05
"""

import os
from typing import Any, Dict, Tuple

import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
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


# tokenize用関数
# tokenizerは関数のスコープ外を参照している点に注意
# ここでpaddingは行わず、トレーニング時にバッチごとにpaddingを行う
def tokenize_function(examples: Dict[str, Any]):
    return tokenizer(examples["sentence"], max_length=512, truncation=True, padding=False, return_tensors=None)  # noqa


def main() -> None:
    # Datasetの読み込み
    train_dataset = Dataset.from_csv(f"{DATA_DIR}/SST-2/train.tsv", sep="\t")
    dev_dataset = Dataset.from_csv(f"{DATA_DIR}/SST-2/dev.tsv", sep="\t")

    # Google BERT
    # https://huggingface.co/google-bert/bert-base-cased
    model_id = "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    # tokenize
    # ここで[CLS], [SEP] tokenも追加される
    encoded_train_dataset = train_dataset.map(
        tokenize_function, batched=True, batch_size=1000, num_proc=2, remove_columns=["sentence"]
    )
    encoded_dev_dataset = dev_dataset.map(
        tokenize_function, batched=True, batch_size=1000, num_proc=2, remove_columns=["sentence"]
    )

    # collate関数, batchごとにPaddingを行う
    # https://huggingface.co/docs/transformers/main_classes/data_collator#transformers.DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir="./results",
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
        fp16=True,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_dev_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_accuracy,
    )

    trainer.train()

    # 最終評価
    eval_results = trainer.evaluate()
    print(f"最終評価結果: {eval_results}")


if __name__ == "__main__":
    main()
