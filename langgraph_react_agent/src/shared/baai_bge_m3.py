from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from langchain_qdrant.sparse_embeddings import SparseVector
from FlagEmbedding import BGEM3FlagModel

# BAAI/bge-m3 模型變數宣告
# 參考 https://huggingface.co/BAAI/bge-m3


class BGEM3QdrantDenseEmbeddings(Embeddings):
    """使用 BAAI/bge-m3 為 Langchain_qdrant 客製的密集向量嵌入模型"""

    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, cache_dir="D:\\model")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, return_dense=True, return_sparse=False)["dense_vecs"]

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query], return_dense=True, return_sparse=False)["dense_vecs"][0]


class BGEM3QdrantSparseEmbeddings(SparseEmbeddings):
    """使用 BAAI/bge-m3 為 Langchain_qdrant 客製的稀疏向量嵌入模型"""

    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, cache_dir="D:\\model")

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        sparse_embeddings = self.model.encode(texts, return_dense=False, return_sparse=True)["lexical_weights"]
        return [
            SparseVector(
                indices=list(map(lambda x: int(x), dict(default_dict).keys())),
                values=list(dict(default_dict).values())
            )
            for default_dict in sparse_embeddings
        ]

    def embed_query(self, query: str) -> SparseVector:
        sparse_embeddings = self.model.encode(
            [query], return_dense=False, return_sparse=True)["lexical_weights"]
        return [
            SparseVector(
                indices=list(map(lambda x: int(x), dict(default_dict).keys())),
                values=list(dict(default_dict).values())
            )
            for default_dict in sparse_embeddings
        ][0]


class BAAIBGEM3Embedding(Embeddings):
    """
      1. BAAI/bge-M3 嵌入模型
      2. embed_documents、embed_query 只是為了 implement 繼承 abstract class Embeddings 的方法，回傳為空陣列
    """

    dense = BGEM3QdrantDenseEmbeddings()
    """密集向量嵌入"""

    sparse = BGEM3QdrantSparseEmbeddings()
    """稀疏向量嵌入"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []
    """請改用 dense.embed_documents 或 sparse.embed_documents"""

    def embed_query(self, query: str) -> list[float]:
        return []
    """請改用 dense.embed_query 或 sparse.embed_query"""
