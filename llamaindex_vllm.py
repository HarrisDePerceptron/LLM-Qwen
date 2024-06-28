"""
First fire up the OpenAI compatible VLLM server 
using the script in the directory
"""

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext

# Set prompt template for generation (optional)


from llama_index.core import StorageContext, load_index_from_storage


from llama_index.llms.openai_like import OpenAILike


def completion_to_prompt(completion):
    return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"


def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system"):
        prompt = "<|im_start|>system\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"

    return prompt


llm = OpenAILike(  # type: ignore
    api_base="http://localhost:8000/v1",
    max_tokens=512,
    temperature=0,
    api_key="EMPTY",
    model="Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
)


Settings.llm = llm
# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Set the size of the text chunk for retrieval
Settings.transformations = [SentenceSplitter(chunk_size=1024)]


documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model,
    transformations=Settings.transformations,
)


# save index
# storage_context = StorageContext.from_defaults(persist_dir="save")

# load index
# index = load_index_from_storage(storage_context)


index.storage_context.persist("./save/vector_store")

query_engine = index.as_query_engine(streaming=True)
your_query = "Qwen 7B memory required"
print(query_engine.query(your_query).print_response_stream())
