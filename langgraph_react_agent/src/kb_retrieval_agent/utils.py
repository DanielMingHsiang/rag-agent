"""Utility & helper functions."""

import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


def get_message_text(msg: BaseMessage) -> str:
    """取得訊息的文字內容

    此函數從各種訊息格式中提取文字內容

    Args:
        msg (AnyMessage): 要從中提取文字的訊息物件

    Returns:
        str: 提取的訊息文字內容

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content={"text": "World"}))
        'World'
        >>> get_message_text(HumanMessage(content=[{"text": "Hello"}, " ", {"text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """從指定的名稱載入聊天模型
    Args:
        fully_specified_name (str): 格式為「provider/model」的字串
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    if provider == "AWS.Bedrock":
        # from langchain_aws import ChatBedrock
        # aws_region = os.environ["AWS_REGION"]
        # bedrock_chat_model = ChatBedrock(
        #     model_id=model,
        #     model_kwargs=dict(temperature=0),
        #     region_name=aws_region,
        # )
        # return bedrock_chat_model
        from langchain_aws import ChatBedrockConverse
        bedrock_chat_model = ChatBedrockConverse(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            temperature=0.0,
            max_tokens=8192,
            # other params...
        )
        return bedrock_chat_model
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url="...")
    else:
        return init_chat_model(model, model_provider=provider)
