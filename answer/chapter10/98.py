"""ファインチューニング.

Refs:
1. https://neptune.ai/blog/fine-tuning-llama-3-with-lora
2. https://github.com/huggingface/peft
3. https://www.reddit.com/r/LocalLLaMA/comments/15sgg4m/what_modules_should_i_target_when_training_using/
"""

import os
from typing import Dict, Tuple

import evaluate
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorWithPadding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def main() -> None:
    # Datasetの読み込み
    train_dataset = Dataset.from_csv(f"{DATA_DIR}/SST-2/train.tsv", sep="\t")
    dev_dataset = Dataset.from_csv(f"{DATA_DIR}/SST-2/dev.tsv", sep="\t")

    # Prepare model, tokenizer and templates
    # Llama-3.2-1B-Instruct
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    # LoRA fine tuning rank=8
    target_modules = ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # Llamaはpad_tokenが設定されていないためEOSで代用 (これでいいのか？)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True, num_labels=2
    )
    model.config.pad_token_id = tokenizer.pad_token_id  # modelにもpadding tokenを設定(Warningの回避)
    model.config.do_sample = False  # greedy method

    # tokenize用関数
    def tokenize_function(examples):
        prompts = []
        for sentence in examples["sentence"]:
            text = f"""You are a classification system for the sentiment classification.
                    For example, if "Input" sentence is the positive sentence like
                    'The movie was full of fun.', you should return label '1' (positive).
                    If "Input" sentence is negative, you should return 0.

                    Please classify following input.

                    Input: {sentence}
                    """
            prompts.append(text)

        # トークン化
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding=False,  # 後でDataCollatorでpadding
            max_length=512,
            return_tensors=None,
        )

        # ラベルを追加
        tokenized["labels"] = examples["label"]

        return tokenized

    # target_modulesに対するLoRAの導入
    model = get_peft_model(model, peft_config)

    # データセットの前処理（トークン化）
    encoded_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])
    encoded_dev_dataset = dev_dataset.map(tokenize_function, batched=True, remove_columns=["sentence"])

    # collate関数 (batchごとにPaddingを行う)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir="./results_98py",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,
        logging_dir="./logs",
        save_only_model=True,  # 学習済モデルのみ保存
    )

    # Training
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
    print(f"Accuracy (dev dataset): {eval_results}")


if __name__ == "__main__":
    main()
