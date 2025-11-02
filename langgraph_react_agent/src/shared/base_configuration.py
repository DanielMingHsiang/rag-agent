"""定義 Agent 的可配置參數。"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar
from langchain_core.runnables import RunnableConfig, ensure_config


@dataclass(kw_only=True)
class BaseConfiguration:
    """基礎 Agent 設定"""

    embedding_model: Annotated[
        Literal[
            "AWS.Bedrock/cohere.embed-multilingual-v3",
            "BAAI/bge-m3",
            "Microsoft/intfloat/multilingual-e5-large",
            "google_genai/gemini-embedding-exp-03-07"
        ],
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="AWS.Bedrock/cohere.embed-multilingual-v3",
        metadata={"description": "嵌入模型的名稱。名稱格式:provider/model-name"},
    )

    query_model: Annotated[
        Literal[
            "AWS.Bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            "AWS.Bedrock/us.amazon.nova-pro-v1:0",
            "AWS.Bedrock/us.meta.llama3-3-70b-instruct-v1:0",
            "AWS.Bedrock/us.deepseek.r1-v1:0",
            "google_genai/gemini-2.0-flash",
            "ollama/gemma3:1b"
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = field(
        default="AWS.Bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        metadata={
            "description": "用於處理和最佳化查詢的 LLM。名稱格式:provider/model-name"
        },
    )

    response_model: Annotated[
        Literal[
            "AWS.Bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            "AWS.Bedrock/us.amazon.nova-pro-v1:0",
            "AWS.Bedrock/us.meta.llama3-3-70b-instruct-v1:0",
            "AWS.Bedrock/us.deepseek.r1-v1:0",
            "google_genai/gemini-2.0-flash",
            "ollama/gemma3:1b"
        ],
        {"__template_metadata__": {"kind": "llm"}},
    ] = field(
        default="AWS.Bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        metadata={
            "description": "用於產生 response 的 LLM。名稱格式:provider/model-name"
        },
    )

    retriever_provider: Annotated[
        Literal["qdrant", "elastic", "elastic-local", "pinecone", "mongodb"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="qdrant",
        metadata={
            "description": """用於檢索的向量儲存提供者。選項包括"qdrant"、"elastic"、"pinecone"或"mongodb"。"""
        },
    )

    document_type: Annotated[
        Literal["insurance", "system_analysis"],
        {"__template_metadata__": {"kind": "document type"}},
    ] = field(
        default="insurance",
        metadata={
            "description": """檢索的文件類型。選項包括"insurance"、"system_analysis"。"""
        },
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "額外的 keyword arguments 用以傳入到 retriever 的 search function"
        },
    )

    retrieve_limit: int = field(
        default=5,
        metadata={"description": "檢索出來的文件筆數上限值"},
    )

    retrieve_filter_key: str = field(
        default="doc_name",
        metadata={"description": "限制檢索範圍的篩選器 key 值"},
    )

    retrieve_filter_text: str = field(
        default="",
        metadata={"description": "限制檢索範圍的篩選器 text 值"},
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=BaseConfiguration)
