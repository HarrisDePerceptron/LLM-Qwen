from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from datetime import datetime

from transformers.generation import streamers
import torch


device = "cuda"  # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
    device_map="auto",
    attn_implementation="flash_attention_2",
    # torch_dtype="auto",
    torch_dtype="auto",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct-GPTQ-Int8")

# Instead of using model.chat(), we directly use model.generate()
# But you need to use tokenizer.apply_chat_template() to format your inputs as shown below
prompt = "why cpu is important"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("chat template: ", text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)


t1 = datetime.now()
# Directly use generate() and tokenizer.decode() to get the output.
# Use `max_new_tokens` to control the maximum output length.


streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generated_ids = model.generate(
    model_inputs.input_ids, max_new_tokens=250, do_sample=False, streamer=streamer
)
# generated_ids = [
#    output_ids[len(input_ids) :]
#    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

t2 = datetime.now()

diff = t2 - t1

print("Took ", diff.total_seconds())
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print("response: ", response)
