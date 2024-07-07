from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.prompts.system import SHAKESPEARE_WRITING_ASSISTANT
from llama_index.core.chat_engine.types import ChatMode


documents = SimpleWebPageReader(html_to_text=True).load_data(
    [
#        "https://www.centrox.ai",
#       "https://www.centrox.io",
#        "https://www.centrox.io/aboutus",
#        "https://www.centrox.ai/contact",
#        "https://www.centrox.ai/team",
        "https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-18,2024-06-18&genres=mystery,thriller",
    ]
)

document_ids = ["1yZ9rLSgp4LkKSq7r_QvbnMwqvxNjuoMcyUxfc3MYS_8"]


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


# llm = OpenAI()
llm = OpenAILike(  # type: ignore
    api_base="http://localhost:8000",
    max_tokens=512,
    temperature=0,
    api_key="EMPTY",
    model="Qwen/Qwen2-1.5B-Instruct",
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
)

llm.is_chat_model = True

data = SimpleDirectoryReader(input_dir="./data").load_data()

data.extend(documents)

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.transformations = [SentenceSplitter(chunk_size=1024)]
Settings.llm = llm

index = VectorStoreIndex.from_documents(
    data, embed_model=Settings.embed_model, transformations=Settings.transformations
)
chat_engine = index.as_chat_engine(
    chat_mode=ChatMode.REACT,
    verbose=True,
    streaming=True,
    llm=llm,
)



while True:
    human = input("Human: ")

    response = chat_engine.stream_chat(human)


    print("AI: ", end="")
    for res in response.response_gen:
        print(res, end="")

    for res in response.chat_stream:
        print(res.message.content, end="")

    print("\n")
