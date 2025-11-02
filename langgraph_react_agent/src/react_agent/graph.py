"""定義客製化的 Reasoning & Action(ReAct) Agent

聊天模型有可以呼叫 tool 的支援

"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model

# Define the function that calls the model


async def call_model(state: State, config: RunnableConfig) -> Dict[str, List[AIMessage]]:
    """呼叫為"Agent"提供服務的 LLM

    函數準備提示、初始化模型並處理回應

    Args:
      state(State):對話的目前狀態
      config(RunnableConfig):模型運行的配置

    Returns:
      dict：包含模型回應訊息的 dict
    """
    configuration = Configuration.from_runnable_config(config)

    # 使用 tool 綁定初始化模型。在此處變更模型或新增更多 tool
    model = load_chat_model(configuration.response_model).bind_tools(TOOLS)

    # format 系統提示。自訂此項以變更 Agent 程式的行為
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=timezone.utc).isoformat()
    )

    # 取得模型回應
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages], config
        ),
    )

    # 處理在 last step 時模型仍想使用 tool 的情況
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="抱歉，我無法在指定的步驟內找到您的問題的答案。",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """根據模型的輸出決定下一個節點

    此函數檢查模型的最後一則訊息是否包含 tool 呼叫

    Args:
        state (State): 對話的當前狀態

    Returns:
        str: 下一個要呼叫的 node 的名稱 ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"預期 output edges 是 AIMessage 型別，但得到 {type(last_message).__name__}"
        )
    # 如果沒有呼叫 tool，就結束
    if not last_message.tool_calls:
        return "__end__"
    # 否則執行呼叫 tool
    return "tools"


# Define a new graph

builder = StateGraph(State, input=InputState, config_schema=Configuration)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))

# 將 `call_model` 設定為進入點，也就是這是第一個被呼叫的 node
builder.add_edge("__start__", "call_model")

# 加入一個 conditional edge 用以決定 `call_model` 的下一步動作
builder.add_conditional_edges(
    "call_model",
    # 在 call_model 完成執行後, 下一個被安排的 node(s) 是基於 route_model_output 的回傳結果
    route_model_output,
)

# 加入一個 `tools` -> `call_model` 的 edge ，這裡形成了一個 cycle，使用 tool 後，我們總是回到 call_model
builder.add_edge("tools", "call_model")

# Compile the builder into an executable graph
# 可以透過新增 interrupt points 更新 state 來自訂此功能
graph = builder.compile(
    interrupt_before=[],  # Add node names here to update state before they're called
    interrupt_after=[],  # Add node names here to update state after they're called
)
graph.name = "ReAct Agent"  # This customizes the name in LangSmith
