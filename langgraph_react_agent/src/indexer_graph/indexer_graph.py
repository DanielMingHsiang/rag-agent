"""This "graph" simply exposes an endpoint for a user to upload docs to be indexed."""

from typing import Optional, Sequence

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


from indexer_graph.configuration import IndexConfiguration
from indexer_graph.state import IndexState
from shared import retrieval


def ensure_docs_have_user_id(docs: Sequence[Document], config: RunnableConfig) -> list[Document]:
    """
    確保所有文件在其元資料中都有一個 user_id。

        docs：要被處理的 Sequence[Document] 物件
        config：包含 user_id 的 RunnableConfig 物件

    Return：
        list[Document]：包含被更新 metadata 的 list[Document] 物件
    """

    # user_id = config["configurable"]["user_id"]

    # return [Document(page_content=doc.page_content, metadata={**doc.metadata, "user_id": user_id}) for doc in docs]
    return [Document(page_content=doc.page_content, metadata={**doc.metadata}) for doc in docs]


async def index_docs(state: IndexState, *, config: Optional[RunnableConfig] = None) -> dict[str, str]:
    """
    使用配置的 retriever 對 state 的文件進行非同步嵌入向量資料庫。
    此功能從 state 獲取 docs，確保它們具有 user_id。
    透過 retriver 將 docs 新增至向量資料庫，然後發出訊號讓 docs 從 state 中刪除。
    
    參數：
        state(IndexState):包含文件和檢索器的目前狀態。
        config (Optional[RunnableConfig]): 動態設定。
    """

    if not config:
        raise ValueError("Configuration required to run index_docs.")
    with retrieval.get_retriever(config) as retriever:
        stamped_docs = ensure_docs_have_user_id(state.docs, config)
        await retriever.aadd_documents(stamped_docs)
    return {"docs": "delete"} # 這步驟會把 decs 從 state 中刪除


# Define a new graph


builder = StateGraph(IndexState, config_schema=IndexConfiguration)
builder.add_node(index_docs)
builder.add_edge("__start__", "index_docs")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
graph.name = "IndexGraph"
