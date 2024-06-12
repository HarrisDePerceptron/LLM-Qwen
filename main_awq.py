from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from datetime import datetime

from transformers.generation import streamers
import torch

from awq import AutoAWQForCausalLM


device = "cuda"  # the device to load the model onto

# Now you do not need to add "trust_remote_code=True"
model = AutoAWQForCausalLM.from_quantized(
    "Qwen/Qwen2-7B-Instruct-AWQ",
    device_map="auto",
    attn_implementation="flash_attention_2",
    fuse_layers=True,
)


t1 = datetime.now()
# Directly use generate() and tokenizer.decode() to get the output.
# Use `max_new_tokens` to control the maximum output length.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct-AWQ")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = (
    "You're standing on the surface of the Earth. "
    "You walk one mile south, one mile west and one mile north. "
    "You end up exactly where you started. Where are you?"
)

chat = [
    {
        "role": "system",
        "content": "You are a concise assistant that helps answer questions.",
    },
    {"role": "user", "content": prompt},
]

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

tokens = tokenizer.apply_chat_template(chat, return_tensors="pt")
tokens = tokens.to("cuda:0")

# Generate output
generation_output = model.generate(
    tokens, streamer=streamer, max_new_tokens=64, eos_token_id=terminators
)

t2 = datetime.now()

diff = t2 - t1

print("Took ", diff.total_seconds())
