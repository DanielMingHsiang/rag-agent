"""
給 rag_graph 使用的各種工具，此模組包含用於處理訊息、文件、以及項目中其他常見的操作

Functions:
    get_message_text: 從各種 human input 的訊息格式中提取文字內容
    format_docs: 將文件內容轉換為 xml 格式的字串
"""

import os
import json
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage


def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message.

    This function extracts the text content from various message formats.

    Args:
        msg (AnyMessage): The message object to extract text from.

    Returns:
        str: The extracted text content of the message.

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


def _format_doc(doc: Document) -> str:
    """格式化從向量資料庫 retrieve 出來的文件為 XML，附加 metadata"""
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())

    if meta:
        meta = f" {meta}"

    return f"""
                        <document{meta}>
                            {doc.page_content}
                        </document>
            """


def format_docs(docs: Optional[list[Document]]) -> str:
    """格式化從向量資料庫 retrieve 出來的文件為 XML"""
    if not docs:
        return "<documents></documents>"

    joined_docs = "\n".join(_format_doc(doc) for doc in docs)
    formatted = f"""
                    <documents>{joined_docs}        </documents>
                """
    return formatted


# 格式化檢索出來的 doc 內容為 json
def format_docs_as_json(docs: Optional[list[Document]]) -> str:
    result_dict = [{"pageContent": doc.page_content, "metadata": doc.metadata} for doc in docs]
    formated_json_docs = json.dumps(result_dict, indent=4, ensure_ascii=False)
    return formated_json_docs


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """從指定的名稱載入聊天模型。參數格式為：「provider/model」的字串。"""

    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    if provider == "AWS.Bedrock":
        from langchain_aws import ChatBedrockConverse
        aws_region = os.environ["AWS_REGION"]
        bedrock_chat_model = ChatBedrockConverse(
            model_id=model,
            temperature=0,
            region_name=aws_region,
        )
        return bedrock_chat_model
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url="http://192.168.56.2:11434")
    else:
        return init_chat_model(model, model_provider=provider)
