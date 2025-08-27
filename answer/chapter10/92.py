"""続きのテキストの予測.

Ref:
1. https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
2. https://huggingface.co/docs/transformers/v4.53.3/en/main_classes/pipelines#transformers.TextGenerationPipeline
3. https://huggingface.co/docs/transformers/v4.53.3/en/generation_strategies
4. https://www.nomuramath.com/kv8wr0mp/
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Specify model
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("The movie was full of", return_tensors="pt", padding=True, truncation=True)

# Greedy method
# https://discuss.huggingface.co/t/scores-in-generate/3450
print("\033[31m" + "Greedy method" + "\033[0m")
with torch.no_grad():
    # config
    max_new_tokens = 10
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        return_dict_in_generate=True,  # 生成されたトークン列のみでなく確率などの情報も返す
        output_scores=True,
    )
    outputs = model.generate(**inputs, generation_config=generation_config)
    greedy_output = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    print(greedy_output[0] + "\n")

print("\033[34m" + "Likelihood of each word" + "\033[0m")
for i, (word, score) in enumerate(zip(greedy_output[0].split(), outputs.scores, strict=False)):
    prob = torch.softmax(score, dim=-1).max()
    print(f"{i}. {word} ->\t prob:{prob:.4f}")
