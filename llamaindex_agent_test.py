from importlib.metadata import metadata
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from llama_index.core.tools import (
    FunctionTool,
    QueryEngineTool,
    ToolMetadata,
    function_tool,
)
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = OpenAILike(  # type: ignore
    api_base="http://localhost:8000/v1/",
    max_tokens=512,
    temperature=0.0,
    api_key="EMPTY",
    model="Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
    # messages_to_prompt=messages_to_prompt,
    # completion_to_prompt=completion_to_prompt,
)

Settings.llm = llm


def load_index(dir):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=dir)
        index = load_index_from_storage(storage_context)
        return index
    except Exception as ex:
        msg = str(ex)


data = SimpleDirectoryReader(input_dir="./data").load_data()

index = VectorStoreIndex.from_documents(data, embed_model=Settings.embed_model)


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def default_func(*arg, **kwargs) -> str:
    """the default tool which is called when no other tool is available to answer the query"""
    return "the system is not able to answer the current query. contact the administrator. end the tool execution"


def addition(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


query_engine = index.as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            description="knowledge base about Centrox company. also has some bench mark. also contains python instruction on some image manipulations",
            name="Centrox",
        ),
    )
]


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=addition)

default_tool = FunctionTool.from_defaults(fn=default_func)
func_tools = [multiply_tool, default_tool, add_tool]

tools = func_tools + query_engine_tools
agent = ReActAgent.from_tools(
    tools,
    llm=llm,
    verbose=True,
)

response = agent.chat("Benchmark for qwen2 7B memory requirement on average")

print("Response: ", response)


# print("Prompts:")
#
# for k, v in agent.get_prompts().items():
#    print(f"prompt[{k}] =  {v.template}")
#
