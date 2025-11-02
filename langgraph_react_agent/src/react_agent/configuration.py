"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

from react_agent import prompts
from shared.base_configuration import BaseConfiguration


@dataclass(kw_only=True)
class Configuration(BaseConfiguration):
    """The configuration for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "與 Agent 互動的系統提示。"
            "此提示為 Agent 設定上下文和行為。"
        },
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> Configuration:
        """從 RunnableConfig 建立一個 Configuration instance 物件"""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
