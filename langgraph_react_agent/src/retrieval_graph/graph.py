"""對話檢索 graph 的主要入口點

此模組定義了對話的核心結構和功能檢索圖表。它包括主圖定義、狀態管理、
以及處理使用者輸入、產生查詢、檢索相關文件，並制定答案。
"""

from datetime import datetime, timezone
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, format_docs_as_json, get_message_text, load_chat_model

from shared import retrieval
from shared.logger import retrieval_graph_logger as logger

# Define the function that calls the model


class SearchQuery(BaseModel):
    """用於檢索文件的查詢條件"""

    query: str


async def generate_query(state: State, *, config: RunnableConfig) -> dict[str, list[str]]:
    """根據目前 state 和 configuration 產生檢索查詢。

    此函數分析 state 中的 messages 並產生適當的"檢索查詢"。對於第一條 message，它直接使用使用者的輸入
    對於後續訊息，它使用 LLM 來產生更細化的"檢索查詢"

    Args:
        state (State): 目前的狀態，包含 messages 和其他資訊.
        config (RunnableConfig | None, optional): 用來產生"檢索查詢"的設定

    Returns:
        dict[str, list[str]]: key 為 'queries'，value 為 "檢索查詢" list 的 dict 物件

    Behavior:
        - 如果只有一條 message（第一個使用者輸入），它會使用該訊息作為查詢。
        - 對於後續訊息，它使用語言模型來產生更細化的"檢索查詢"
        - 此函數使用 configuration 來設定 prompt 和模型
    """

    configuration = Configuration.from_runnable_config(config)
    thread_id = config["metadata"]["thread_id"]
    logger.info(f"""{"==="*15} [start][thread_id={thread_id}] {"==="*15}""")
    logger.info("[查詢] LLM：%s", configuration.query_model)
    logger.info("[回應] LLM：%s", configuration.response_model)
    logger.info("Embedding 模型：%s", configuration.embedding_model)
    logger.info("向量資料庫：%s", configuration.retriever_provider)
    logger.info("檢索文件種類：%s", configuration.document_type)
    logger.info("檢索文件篩選 key：%s, text:%s", configuration.retrieve_filter_key, configuration.retrieve_filter_text)

    messages = state.messages
    logger.info("人類 -> %s", messages[-1].content)
    if len(messages) == 1:
        # It's the first user question. We will use the input directly to search.
        human_input = get_message_text(messages[-1])
        logger.info("查詢條件 -> %s", [human_input])
        return {"queries": [human_input]}
    else:
        configuration = Configuration.from_runnable_config(config)
        # Feel free to customize the prompt, model, and other logic!
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        model = load_chat_model(configuration.query_model).with_structured_output(SearchQuery)

        message_value = await prompt.ainvoke(
            {
                "messages": state.messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )

        generated = cast(SearchQuery, await model.ainvoke(message_value, config))

        logger.info("查詢條件 -> %s", generated.query)
        return {"queries": [generated.query]}


async def retrieve(state: State, *, config: RunnableConfig) -> dict[str, list[Document]]:
    """根據 state 裡最新的 user query 檢索文件。

    此函數取得目前 state 和 config，使用最新的 query 檢索相關文檔，並返回檢索到的文檔。

    Args:
        state (State): 包含 queries 和 retriever.
        config (RunnableConfig | None, optional): 檢索過程中使用到的設定

    Returns:
        dict[str, list[Document]]: 包含單一 key -> "retrieved_docs" 的 dicti 物件，內容為 Document 陣列物件
    """
    with retrieval.get_retriever(config) as retriever:
        response = await retriever.ainvoke(state.queries[-1], config)
        return {"retrieved_docs": response}


async def respond(state: State, *, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """呼叫 LLM 提供 "Agent" 執行程序"""
    configuration = Configuration.from_runnable_config(config)
    # Feel free to customize the prompt, model, and other logic!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("placeholder", "{messages}"),
        ]
    )
    model = load_chat_model(configuration.response_model)

    retrieved_docs = format_docs_as_json(state.retrieved_docs)
    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)

    logger.info("LLM 回應 -> %s", str(response.content))
    logger.info("檢索文件  -> \n%s", retrieved_docs)
    thread_id = config["metadata"]["thread_id"]
    logger.info(f"""{"==="*15} [ end ][thread_id={thread_id}] {"==="*15}""")

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph (It's just a pipe)


builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node(generate_query)
builder.add_node(retrieve)
builder.add_node(respond)
builder.add_edge("__start__", "generate_query")
builder.add_edge("generate_query", "retrieve")
builder.add_edge("retrieve", "respond")

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
memory = MemorySaver()
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
    # checkpointer=memory,
)
graph.name = "Graph"
