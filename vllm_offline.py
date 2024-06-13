from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from datetime import datetime

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct-AWQ")

# Pass the default decoding hyperparameters of Qwen2-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.05,
    max_tokens=200,
)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="Qwen/Qwen2-7B-Instruct-AWQ")

# Prepare your prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# messages = [
#    {"role": "system", "content": "You are a helpful assistant."},
#    {"role": "user", "content": prompt},
# ]
# text = tokenizer.apply_chat_template(
#    messages, tokenize=False, add_generation_prompt=True
# )
t1 = datetime.now()
# generate outputs
# outputs = llm.generate([text], sampling_params)
#
outputs = llm.generate(prompts, sampling_params)

t2 = datetime.now()

diff = t2 - t1
took = diff.total_seconds()


print(f"Took {took}ms")


# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
