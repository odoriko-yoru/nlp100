"""マルチターンのチャット.

1. https://huggingface.co/learn/llm-course/chapter11/2
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Prepare model, tokenizer and templates
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)

generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a doctor. \
        Answer questions directly and accurately.",
    },
    {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
]

# 最初の会話
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(tokenized_chat, generation_config=generation_config, max_new_tokens=50)
first_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 応答
first_answer = first_response.split("assistant")[-1].strip()

# 応答をmessagesに追加
messages.append({"role": "assistant", "content": first_answer})

# 追加質問
messages.append(
    {"role": "user", "content": "Please give me the plural form of the word with its spelling in reverse order."}
)

# 応答
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(tokenized_chat, generation_config=generation_config, max_new_tokens=50)
second_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("=== 応答1 ===")
print("質問: What do you call a sweet eaten after dinner?")
print(f"回答: {first_answer}\n")
print("=== 応答2 ===")
print("質問: Please give me the plural form of the word with its spelling in reverse order.")
print(f"回答: {second_response.split('assistant')[-1].strip()}")
