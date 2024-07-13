from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from torch import transpose
from llama_index.core.agent import ReActAgent
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


Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Settings.transformations = [SentenceSplitter(chunk_size=1024)]


llm = OpenAILike(  # type: ignore
    api_base="http://localhost:8000/v1/",
    max_tokens=1024,
    temperature=0.5,
    api_key="EMPTY",
    model="Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
)

# llm.is_chat_model = True

Settings.llm = llm

try:
    storage_context = StorageContext.from_defaults(persist_dir="./save/lyft")
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(persist_dir="./save/uber")
    uber_index = load_index_from_storage(storage_context)

    index_loaded = True
except Exception as ex:
    msg = str(ex)
    index_loaded = False


if not index_loaded:
    # load data
    lyft_docs = SimpleDirectoryReader(
        input_files=["./data/10k/lyft_2021.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=["./data/10k/uber_2021.pdf"]
    ).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(
        lyft_docs,
        embed_model=Settings.embed_model,
        transformations=Settings.transformations,
    )
    uber_index = VectorStoreIndex.from_documents(
        uber_docs,
        embed_model=Settings.embed_model,
        transformations=Settings.transformations,
    )

    # persist index
    lyft_index.storage_context.persist(persist_dir="./save/lyft")
    uber_index.storage_context.persist(persist_dir="./save/uber")

lyft_engine = lyft_index.as_query_engine()
uber_engine = uber_index.as_query_engine()


query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    # context=context
)

response = agent.chat("What was Lyft's revenue growth in 2021?")
print(str(response))


response = agent.chat(
    "Compare and contrast the revenue growth of Uber and Lyft in 2021, then"
    " give an analysis"
)
print(str(response))


response = agent.query(
    "Can you tell me about the risk factors of the company with the higher revenue?"
)
print(str(response))
