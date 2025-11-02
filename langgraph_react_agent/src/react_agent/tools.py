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
from typing_extensions import Annotated

from shared import retrieval
from react_agent.configuration import Configuration


@tool
async def search(query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.

    Args:
        query (str): search query
    Returns:
        Optional[list[dict[str, Any]]]: list of search result
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    print("query", query)
    print("configuration", configuration)
    return cast(list[dict[str, Any]], result)


@tool
async def retrieve(
    query: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """檢索知識庫提供 LLM 的參考回應"""
    with retrieval.get_retriever(config) as retriever:
        try:
            response = await retriever.ainvoke(query, config)
        except Exception as e:
            print(e)
        # return cast(list[dict[str, Any]], response)
        return [{"link": "https://123456", "doc_name": doc.metadata["doc_name"], "pageContent": doc.page_content, "metadata": doc.metadata} for doc in response]


@tool
async def retrieve_ctbc_sa_doc(
    # query:str,
    taskId: Annotated[str, {"__tool_param__": {"kind": "功能代碼"}}] = field(
        default=None,
        metadata={"description": "功能代碼，例如 TWRBM_XXX_XXX"},
    ),
    # description: Annotated[str, {"__tool_param__": {"kind": "查詢描述"}}] = field(
    #     default=None,
    #     metadata={"description": "查詢的內容，例如 moc 使用了那些"},
    # ),
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """檢索知識庫中的規格書相關資訊

    必須取得 taskId 與 description 參數，缺一不可
    使用 description 參數做為知識庫檢索的查詢條件
    使用 taskId 做為檢索出的文件的過濾篩選條件

    Args:
        taskId (str): 功能代碼
    Returns:
        str: 規格書的描述

    """
    # if location.lower() in ["sf", "san francisco", "my home"]:
    #     return "It's 60 degrees and foggy."
    # else:
    #     return "It's 90 degrees and sunny."
    configuration = Configuration.from_runnable_config(config)
    # print("taskId", query)
    # print("taskId", taskId)
    # print("description", description)
    print("configuration", configuration)
    return "資訊太多尚無法回答"


TOOLS: List[Callable[..., Any]] = [retrieve]
