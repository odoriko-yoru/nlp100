"""続きのテキストの予測.

Ref:
1. https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
2. https://huggingface.co/docs/transformers/v4.53.3/en/main_classes/pipelines#transformers.TextGenerationPipeline
3. https://huggingface.co/docs/transformers/v4.53.3/en/generation_strategies
4. https://www.nomuramath.com/kv8wr0mp/
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Specify model
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)


inputs = tokenizer("The movie was full of", return_tensors="pt", padding=True, truncation=True)

# config
tempratures = [1.0, 0.7, 0.5, 0.1]
max_new_tokens = 50

# Basic Decoding method
# Greedy method
print("\033[31m" + "Greedy method" + "\033[0m")
generation_config = GenerationConfig(
    max_new_tokens=max_new_tokens,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False,
)
outputs = model.generate(**inputs, generation_config=generation_config)
greedy_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(greedy_output[0] + "\n")

# Beam search
print("\033[31m" + "BeamSearch (num_beam = 5)" + "\033[0m")
generation_config = GenerationConfig(
    max_new_tokens=max_new_tokens,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=False,
    num_beams=5,
)
outputs = model.generate(**inputs, generation_config=generation_config)
sampling_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(sampling_output[0] + "\n")


# Sampling
print("\033[31m" + "Multinomial Sampling" + "\033[0m")
for t in tempratures:
    print(f"Temperature = {t}")
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        num_beams=1,
        temperature=t,
    )
    outputs = model.generate(**inputs, generation_config=generation_config)
    sampling_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(sampling_output[0] + "\n")
