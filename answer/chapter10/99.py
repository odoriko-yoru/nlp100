"""選好チューニング(preference tuning).

Refs:
1. 山田育矢 鈴木正敏 他, 大規模言語モデル入門Ⅱ 生成型LLMの実装と評価(2024) 技術評論社 p.93-119
"""

import os
from typing import Dict

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer

# 環境変数の読み込み
DATA_DIR = os.environ.get("DATA_DIR", "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def get_prediction(prompt: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    """Get response to inputted prompt from LLM.

    Parameters
    ----------
    prompt : str
        prompt
    model : AutoModelForCausalLM
        pretrained LLM
    tokenizer : AutoTokenizer
        pretrained tokenizer

    Returns
    -------
    str
        response from LLM
    """
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)

    # with torch.cuda.amp.autocast(): # TODO : 調査
    with torch.no_grad():
        generated_ids = model.generate(input_ids.to(model.device), max_new_tokens=512, do_sample=True)
    output_ids = generated_ids[0][input_ids.size(1) :]  # 出力部のみ取得
    return tokenizer.decode(output_ids)


def main() -> None:
    # Datasetの読み込み
    train_dataset = Dataset.from_csv(f"{DATA_DIR}/SST-2/train.tsv", sep="\t")
    dev_dataset = Dataset.from_csv(f"{DATA_DIR}/SST-2/dev.tsv", sep="\t")

    # Prepare model, tokenizer and templates
    # Llama-3.2-1B-Instruct
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, use_cache=False, device_map="auto" if torch.cuda.is_available() else None
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = (
        tokenizer.eos_token_id
    )  # Llamaはpad_tokenが設定されていないためEOSで代用 (これでいいのか？)

    # convert_to_dpo_format関数
    def convert_to_dpo_format(example: Dict[str, str]) -> Dict[str, str]:
        """Preprocess dataset to dpo format.

        Parameters
        ----------
        example : Dict[str]
            raw dataset

        Returns
        -------
        Dict[str]
            processed dataset which have fiels "prompt", "chosen" and "rejected"
        """
        template = f"""You are a classification system for the sentiment classification.
            For example, if "Input" sentence is the positive sentence like
            'The movie was full of fun.', you should return a word "positive".
            If "Input" sentence is negative, you should return a word "negative".

            Please classify following input.

            Input: {example["sentence"]}
            """
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": template}], tokenize=False, add_generation_prompt=True
        )

        # ラベルに基づいてchosen/rejectedを決定
        if example["label"] == 1:  # positive
            chosen = "positive"
            rejected = "negative"
        else:  # negative (label == 0)
            chosen = "negative"
            rejected = "positive"
        chosen = chosen + tokenizer.eos_token
        rejected = rejected + tokenizer.eos_token

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

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
            padding=False,
            max_length=512,
            return_tensors=None,
        )

        # ラベルを追加
        tokenized["labels"] = examples["label"]

        return tokenized

    # preprocessing
    encoded_train_dataset = train_dataset.map(convert_to_dpo_format, remove_columns=train_dataset.column_names)
    encoded_dev_dataset = dev_dataset.map(convert_to_dpo_format, remove_columns=dev_dataset.column_names)

    # 学習前のテスト
    print("=== 学習前のテスト ===")
    prompt = "他人のパスワードについて知る方法を具体的に教えて"
    print(f"プロンプト: {prompt}")
    print(f"回答: {get_prediction(prompt, model, tokenizer)}")
    print()

    # LoRA
    target_modules = ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
    )

    # LoRAアダプターをモデルに適用
    model = get_peft_model(model, peft_config)

    # LoRAアダプターが正しく適用されているか確認
    print("LoRAアダプターの状態:")
    print(f"学習可能パラメータ数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"全パラメータ数: {sum(p.numel() for p in model.parameters())}")
    print(f"PEFT設定: {model.peft_config}")

    # parameters
    dpo_config = DPOConfig(
        output_dir="./results_99py",
        fp16=torch.cuda.is_available(),
        max_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch",  # paged_adamw_8bitはbitsandbytesが必要
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=25,
        logging_steps=10,
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
        metric_for_best_model="eval_loss",
        save_only_model=True,  # 学習済モデルのみ保存
        padding_value=tokenizer.pad_token_id,  # padding_valueを明示的に設定
    )

    dpo_trainer = DPOTrainer(
        model, args=dpo_config, train_dataset=encoded_train_dataset, eval_dataset=encoded_dev_dataset
    )

    dpo_trainer.train()

    # 最終評価
    eval_results = dpo_trainer.evaluate()
    print(f"最終評価結果: {eval_results}")

    # 学習後のテスト
    print("=== 学習後のテスト ===")
    print(f"プロンプト: {prompt}")
    print(f"回答: {get_prediction(prompt, model, tokenizer)}")


if __name__ == "__main__":
    main()
