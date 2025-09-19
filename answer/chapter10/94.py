"""チャットテンプレート.

Ref.
1. https://huggingface.co/docs/transformers/main/chat_templating
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
        "content": "You are a friendly chatbot who always responds in the style of a medical doctor. \
        Answer questions directly and accurately.",
    },
    {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
]

tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")

outputs = model.generate(tokenized_chat, generation_config=generation_config, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant")[-1].strip())
