"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from dataclasses import field
from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langchain_core.tools import tool
from langchain_core.documents import Document

from pydantic import BaseModel
from typing_extensions import Annotated

from shared import retrieval
from kb_retrieval_agent.configuration import Configuration
from shared.base_configuration import BaseConfiguration


# 中途 AI 詢問人類的時候要用以下格式回應
# {
#   "messages": [
#     {
#       "tool_call_id": "",
#       "type": "tool",
#       "content": "功能名稱是歷史明細"
#     }
#   ]
# }

class AskHuman(BaseModel):
    """詢問使用者相關的必要資訊"""

    question: str


@tool
async def retrieve_insurance_doc(
    query: Annotated[str, {"__tool_param__": {"kind": "查詢條件"}}] = field(
        metadata={"description": "查詢條件"},
    ),
    age: Annotated[str, {"__tool_param__": {"kind": "年齡"}}] = field(
        metadata={"description": "年齡"},
    ),
    gender: Annotated[str, {"__tool_param__": {"kind": "性別"}}] = field(
        metadata={"description": "性別"},
    ),
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """檢索知識庫中的保險相關資訊

    使用 query、age、gender 做為知識庫檢索的查詢條件

    Args:
        query (str): 使用者問題，必要參數
        age (str): 年齡
        gender (str): 性別
    Returns:
        Optional[list[dict[str, Any]]]: 檢索知識庫取得的文件清單

    """
    config.setdefault("document_type", "insurance")
    configuration = BaseConfiguration.from_runnable_config(config)
    with retrieval.get_retriever(config) as retriever:
        queryStr = f"{query}, 年齡: {age}, 性別:{gender}"
        response = await retriever.ainvoke(queryStr, RunnableConfig(
            configurable={
                "search_kwargs": {
                    "k": configuration.retrieve_limit,
                    "filter": None
                },
            }
        ))
        if len(response) == 0:
            return "無搜尋到相關資訊"
        else:
            return [{"doc_name": doc.metadata["doc_name"], "pageContent": doc.page_content, "metadata": doc.metadata} for doc in response]


@tool
async def retrieve_ctbc_sa_doc(
    # task_id: Annotated[str, {"__tool_param__": {"kind": "功能代碼"}}] = field(
    #     metadata={"description": """功能代碼，例如 "TWRBM_XXX_XXX"、"TWRBC_XXX_XXX" """},
    # ),
    task_name: Annotated[str, {"__tool_param__": {"kind": "規格書名稱"}}] = field(
        metadata={"description": """規格書名稱，例如"台幣轉帳"、"台幣存款概要"、"基金申購"等等"""},
    ),
    search_content: Annotated[str, {"__tool_param__": {"kind": "搜尋內容"}}] = field(
        metadata={"description": "搜尋內容。例如描述、功能設計概要、電文規格清單等等"},
    ),
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """檢索知識庫中的規格書相關資訊

    使用 search_content 做為知識庫檢索的查詢條件
    使用 task_id 或 task_name 做為檢索出文件的過濾篩選條件

    Args:
        task_name (str): 規格書名稱
        search_content (str): 搜尋內容，必要參數
    Returns:
        Optional[list[dict[str, Any]]]: 檢索知識庫取得的文件清單

    """
    config.setdefault("document_type", "system_analysis")
    configuration = BaseConfiguration.from_runnable_config(config)
    with retrieval.get_retriever(config) as retriever:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
        should_condition = []
        # if task_id:
        #     should_condition.append(FieldCondition(
        #         key=f"metadata.doc_name",
        #         match=MatchText(text=task_id)
        #     ))
        if task_name:
            should_condition.append(FieldCondition(
                key=f"metadata.doc_name",
                match=MatchText(text=task_name)
            ))
        try:
            response = await retriever.ainvoke(
            search_content,
            RunnableConfig(
                configurable={
                    "search_kwargs": {
                        "k": configuration.retrieve_limit,
                        "filter": Filter(
                            should=should_condition,
                        ),
                    },
                }
            )
        )
        except Exception as e:
            print(e)
        
        if len(response) == 0:
            return "無搜尋到相關資訊"
        else:
            return [{"link": "https://123456", "doc_name": doc.metadata["doc_name"], "pageContent": doc.page_content, "metadata": doc.metadata} for doc in response]

        # modify_config = RunnableConfig(
        #     configurable={
        #         "search_kwargs": {
        #             "filter": Filter(
        #                 should=should_condition,
        #             ),
        #         },
        #     }
        # )
        # return cast(list[Document], response)
        # return {"retrieved_docs": response}


TOOLS: List[Callable[..., Any]] = [
    retrieve_insurance_doc,
    retrieve_ctbc_sa_doc,
    # AskHuman
]
