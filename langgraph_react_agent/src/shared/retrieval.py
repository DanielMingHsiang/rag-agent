"""
管理各種 retriver 的配置。

此模組提供建立和管理不同 retriver、向量儲存後端的功能

檢索器支援透過 user_id 過濾結果，確保使用者之間的資料隔離。
"""

import os
from contextlib import contextmanager
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig, ConfigurableField
from langchain_core.vectorstores import VectorStoreRetriever

from shared.base_configuration import BaseConfiguration


@contextmanager
def get_retriever(config: RunnableConfig) -> Generator[VectorStoreRetriever, None, None]:
    """根據目前 configuration，為 agent 建立一個 retriever。"""

    configuration = BaseConfiguration.from_runnable_config(config)

    embedding_model = get_match_embedding(configuration.embedding_model)

    match configuration.retriever_provider:
        case "qdrant":
            with get_qdrant_retriever(configuration, embedding_model) as retriever:
                yield retriever
        # case "mongodb":
        #     with make_mongodb_retriever(configuration, embedding_model) as retriever:
        #         yield retriever
        case _:
            raise ValueError(
                "無法辨識的 retriever provider。"
                f"應為以下幾種: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"但要求: {configuration.retriever_provider}"
            )


def get_match_embedding(model: str) -> Embeddings:
    """取得對應的 Embedding 模型"""
    provider, model = model.split("/", maxsplit=1)
    huggingface_cache_folder = os.environ["HUGGINGFACE_CACHE_FOLDER"]
    match provider:
        case "AWS.Bedrock":
            from langchain_aws import BedrockEmbeddings
            aws_region = os.environ["AWS_REGION"]
            return BedrockEmbeddings(region_name=aws_region, model_id=model)
        case "BAAI":
            from shared.baai_bge_m3 import BAAIBGEM3Embedding
            return BAAIBGEM3Embedding()
        case "Microsoft":
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model, cache_folder=huggingface_cache_folder)
        case "google_genai":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=f"models/{model}")
        case _:
            raise ValueError(f"不支援的 embedding provider: {provider}")


# ===== get retriver 區塊 ================================================
@contextmanager
def get_qdrant_retriever(configuration: BaseConfiguration, embedding_model: Embeddings) -> Generator[VectorStoreRetriever, None, None]:
    """設定此 agent 連線到特定 Qdrant collection"""
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client.http.models import Distance

    fully_specified_name = configuration.embedding_model
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_api_key = os.environ["QDRANT_API_KEY"]
    print('qdrant_url',f'[{qdrant_url}]')
    print('qdrant_api_key',f'[{qdrant_api_key}]')
    qdrant_collection_name = get_qdrant_collection_name(provider, configuration.document_type)
    print('qdrant_collection_name',f'[{qdrant_collection_name}]')

    vstore = None
    match provider:
        case "AWS.Bedrock" | "Microsoft":
            vstore = QdrantVectorStore.from_existing_collection(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=qdrant_collection_name,
                embedding=embedding_model,
                vector_name="dense_text",
                # Euclidean distance，歐氏距離 (L2), Inner Product，內積 (IP), Cosine Similarity，餘弦相似性 (COSINE),
                distance="Euclid",
                retrieval_mode=RetrievalMode.DENSE,
            )
        case "google_genai":
            vstore = QdrantVectorStore.from_existing_collection(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=qdrant_collection_name,
                embedding=embedding_model,
                vector_name="dense_text",
                # Euclidean distance，歐氏距離 (L2), Inner Product，內積 (IP), Cosine Similarity，餘弦相似性 (COSINE),
                distance=Distance.EUCLID,
                retrieval_mode=RetrievalMode.DENSE,
            )
        case "BAAI":
            vstore = QdrantVectorStore.from_existing_collection(
                url=qdrant_url,
                api_key=qdrant_api_key,
                collection_name=qdrant_collection_name,
                # 密集向量區
                embedding=embedding_model.dense,
                vector_name="dense_text",
                # Euclidean distance，歐氏距離 (L2), Inner Product，內積 (IP), Cosine Similarity，餘弦相似性 (COSINE),
                distance="Euclid",
                # /密集向量區
                # 稀疏向量區
                sparse_embedding=embedding_model.sparse,
                sparse_vector_name="sparse_text",
                # /稀疏向量區
                retrieval_mode=RetrievalMode.HYBRID,  # 混合檢索，必須搭配密集+稀疏向量
            )
        case _:
            raise ValueError(f"不支援的 embedding provider: {provider}")

    search_kwargs = configuration.search_kwargs
    search_kwargs.setdefault("k", configuration.retrieve_limit)

    # 設定 seach keyword arguments 篩選檢索結果
    # from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
    # search_kwargs.setdefault(
    #     "filter",
    #     Filter(
    #         should=[
    #             FieldCondition(
    #                 key=f"metadata.{configuration.retrieve_filter_key}",
    #                 match=MatchText(text=configuration.retrieve_filter_text)
    #             )
    #         ],
    #     ),
    # )

    # 這裡將回傳的 retriever 設定為可動態設置的
    yield vstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs).configurable_fields(
        search_kwargs=ConfigurableField(
            id="search_kwargs",
            name="Search Kwargs",
            description="設定 search_kwargs 參數可以動態調整，可配合 filter 篩選 metadata、k 設定選取筆數",
        )
    )


def get_qdrant_collection_name(provider: str, doc_type: str):
    if provider == "AWS.Bedrock":
        match doc_type:
            case "insurance":
                return os.environ["QDRANT_COLLECTION_COHERE_MULTILINGUAL_V3_AWS_EC2"]
            case "system_analysis":
                return os.environ["QDRANT_COLLECTION_COHERE_MULTILINGUAL_V3_SA"]
            case _:
                return os.environ["QDRANT_COLLECTION_COHERE_MULTILINGUAL_V3_AWS_EC2"]
    elif provider == "BAAI":
        match doc_type:
            case "insurance":
                return os.environ["QDRANT_COLLECTION_BAAI_BGEM3_AWS_EC2"]
            case "system_analysis":
                return os.environ["QDRANT_COLLECTION_BAAI_BGEM3_SA"]
            case _:
                return os.environ["QDRANT_COLLECTION_BAAI_BGEM3_AWS_EC2"]
    elif provider == "Microsoft":
        match doc_type:
            case "insurance":
                return os.environ["QDRANT_COLLECTION_MICROSOFT_E5_LARGE_AWS_EC2"]
            case "system_analysis":
                return os.environ["QDRANT_COLLECTION_MICROSOFT_E5_LARGE_SA"]
            case _:
                return os.environ["QDRANT_COLLECTION_MICROSOFT_E5_LARGE_AWS_EC2"]
    elif provider == "google_genai":
        match doc_type:
            case "insurance":
                return os.environ["QDRANT_COLLECTION_GEMINI_EXP_03_07_AWS_EC2"]
            case "system_analysis":
                return os.environ["QDRANT_COLLECTION_GEMINI_EXP_03_07_SA"]
            case _:
                return os.environ["QDRANT_COLLECTION_MICROSOFT_E5_LARGE_AWS_EC2"]
    else:
        raise ValueError(f"不支援的 embedding provider: {provider}")

# @contextmanager
# def make_mongodb_retriever(configuration: IndexConfiguration, embedding_model: Embeddings) -> Generator[VectorStoreRetriever, None, None]:
#     """Configure this agent to connect to a specific MongoDB Atlas index & namespaces."""
#     from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

#     vstore = MongoDBAtlasVectorSearch.from_connection_string(
#         os.environ["MONGODB_URI"],
#         namespace="langgraph_retrieval_agent.default",
#         embedding=embedding_model,
#     )
#     search_kwargs = configuration.search_kwargs
#     pre_filter = search_kwargs.setdefault("pre_filter", {})
#     pre_filter["user_id"] = {"$eq": configuration.user_id}
#     yield vstore.as_retriever(search_kwargs=search_kwargs)
