from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

import gc


def load_model(model_name, flash_attn2=False):
    device = "cuda"  # the device to load the model onto

    args = {"device_map": "cuda:0", "torch_dtype": "auto"}

    if flash_attn2:
        args["attn_implementation"] = "flash_attention_2"

    # Now you do not need to add "trust_remote_code=True"
    model = AutoModelForCausalLM.from_pretrained(model_name, **args)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def run_inference(model, tokenizer, prompt: str):
    device = "cuda"  # the device to load the model onto

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids, max_new_tokens=200, do_sample=False
    )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def main():
    args = [{"flash_attn2": True}, {"flash_attn2": False}]
    prompt = (
        "explain computer cpu. my name is harris and i am a computer science student"
    )
    model_list = [
        "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct-AWQ",
        "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2-1.5B-Instruct-AWQ",
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
        "Qwen/Qwen2-0.5B-Instruct-AWQ",
    ]

    b_f = open("benchmark.txt", "w")

    for model_name in model_list:
        for arg in args:
            model, tokenizer = load_model(model_name, **arg)
            memory = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            allocated_memory = torch.cuda.memory_allocated(0) / (1024 * 1024)
            for i in range(2):
                t1 = datetime.now()
                run_inference(model, tokenizer, prompt)
                t2 = datetime.now()
                diff = t2 - t1
                took = diff.total_seconds()

                msg = f"{i} {model_name}, {str(arg)}, {allocated_memory}mem {took}s\n"

                b_f.write(msg)
                b_f.flush()

                print("Args: ", arg, " Took ", diff.total_seconds())

            model.cpu()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    b_f.close()


if __name__ == "__main__":
    main()
