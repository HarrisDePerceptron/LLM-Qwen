import os
import sys
import logging


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever, RouterRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from llama_index.core.tools import RetrieverTool

from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.postprocessor.colbert_rerank import ColbertRerank

from llama_index.core.retrievers import QueryFusionRetriever

from llama_index.core.query_engine import SubQuestionQueryEngine

from llama_index.core.tools import QueryEngineTool, ToolMetadata


documents = SimpleDirectoryReader("./data/paul_graham").load_data()
documents2 = SimpleDirectoryReader("./data").load_data()


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
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

splitter = SentenceSplitter(chunk_size=250)
nodes = splitter.get_nodes_from_documents(documents)
nodes2 = splitter.get_nodes_from_documents(documents2)

nodes.extend(nodes2)

storage_context = StorageContext.from_defaults()


storage_context.docstore.add_documents(nodes)


index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

vector_retriever = VectorIndexRetriever(index, similarity_top_k=5)

bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

# retriever_tools = [
#    RetrieverTool.from_defaults(
#        retriever=vector_retriever,
#        description="Useful in most cases",
#    ),
#    RetrieverTool.from_defaults(
#        retriever=bm25_retriever,
#        description="Useful if searching about specific information",
#    ),
# ]


# retriever = RouterRetriever.from_defaults(
#    retriever_tools=retriever_tools,
#    llm=llm,
#    select_multi=True,
# )


colbert_reranker = ColbertRerank(
    top_n=5,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=4,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    # mode="relative_score",
    # mode="dist_based_score",
    use_async=True,
    verbose=True,
    # query_gen_prompt="...",  # we could override the query generation prompt here
)
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever, llm=llm, node_postprocessors=[colbert_reranker]
)


# query_engine_tools = [
#    QueryEngineTool(
#        query_engine=query_engine_ret,
#        metadata=ToolMetadata(
#            name="knowledgebase",
#            description="knowledge base to be used for answering every question",
#        ),
#    ),
# ]
#
# query_engine = SubQuestionQueryEngine.from_defaults(
#    query_engine_tools=query_engine_tools,
#    use_async=True,
# )


response = query_engine.query("which school did Paul attend")


for node in response.source_nodes:
    print(node.id_)
    print(node.node.get_content()[:])
    print("reranking score: ", node.score)
    print("retrieval score: ", node.node.metadata["retrieval_score"])
    print("**********")

print(response)


while True:
    print("Human: ", end="")
    q = input()
    r = query_engine.query(q)
    print("AI: ", r)
