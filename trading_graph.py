"""
TradingGraph: Orchestrates the multi-agent trading system using LangChain and LangGraph.
Initializes LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
"""

import math
import os
from typing import Dict
from urllib.parse import urlsplit, urlunsplit

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from langgraph.prebuilt import ToolNode

from env_config import BOB_API_DEFAULT_BASE_URL, BOB_API_DEFAULT_MODEL, build_config
from graph_setup import SetGraph
from graph_util import TechnicalTools


SUPPORTED_PROVIDERS = ("openai", "anthropic", "qwen", "minimax", "minimax_cn", "gemini", "deepseek", "bob-api")
BOB_API_PROVIDER_CONFIG = {
    "label": "Bob API",
    "config_key": "bob_api_key",
    "env_key": "BOB_API_KEY",
    "base_url": BOB_API_DEFAULT_BASE_URL,
    "default_model": BOB_API_DEFAULT_MODEL,
    "timeout": 60,
    "max_retries": 2,
}
MINIMAX_PROVIDER_CONFIG = {
    "minimax": {
        "label": "MiniMax",
        "config_key": "minimax_api_key",
        "env_keys": ("MINIMAX_API_KEY",),
        "base_url": "https://api.minimax.io/v1",
        "console_url": "https://platform.minimaxi.com/",
    },
    "minimax_cn": {
        "label": "MiniMax CN",
        "config_key": "minimax_cn_api_key",
        "env_keys": ("MINIMAX_CN_API_KEY", "MINIMAX_API_KEY"),
        "base_url": "https://api.minimaxi.com/v1",
        "console_url": "https://platform.minimaxi.com/",
    },
}


def normalize_bob_api_base_url(base_url: str = None) -> str:
    """Validate a Bob API endpoint and normalize its OpenAI-compatible /v1 path."""
    value = base_url or BOB_API_PROVIDER_CONFIG["base_url"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Bob API base URL must be a non-empty string")
    parsed = urlsplit(value.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Bob API base URL must be a valid HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("Bob API base URL must not contain a query string or fragment")

    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class TradingGraph:
    """
    Main orchestrator for the multi-agent trading system.
    Sets up LLMs, toolkits, and agent nodes for indicator, pattern, and trend analysis.
    """

    def __init__(self, config=None):
        # --- Configuration and LLMs ---
        # 默认配置由 .env / 环境变量提供（见 env_config.build_config）
        self.config = config if config is not None else build_config()

        # Initialize LLMs with provider support（每个槽位可独立配置 provider/base_url/key/model）
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "openai"),
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
            api_key=self.config.get("agent_llm_api_key"),
            base_url=self.config.get("agent_llm_base_url"),
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "openai"),
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
            api_key=self.config.get("graph_llm_api_key"),
            base_url=self.config.get("graph_llm_base_url"),
        )
        self.toolkit = TechnicalTools()

        # --- Create tool nodes for each agent ---
        # self.tool_nodes = self._set_tool_nodes()

        # --- Graph logic and setup ---
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
            # self.tool_nodes,
        )

        # --- The main LangGraph graph object ---
        self.graph = self.graph_setup.set_graph()

    def _get_api_key(self, provider: str = "openai") -> str:
        """
        Get API key with proper validation and error handling.
        
        Args:
            provider: The provider name ("openai", "anthropic", "qwen", "minimax", "minimax_cn", "gemini" or "deepseek")
        
        Returns:
            str: The API key for the specified provider
            
        Raises:
            ValueError: If API key is missing or invalid
        """
        # 通用 Key（.env 中的 LLM_API_KEY）作为兜底
        generic_key = self.config.get("llm_api_key") or os.environ.get("LLM_API_KEY")

        if provider == "openai":
            # First check if API key is provided in config
            api_key = self.config.get("api_key")
            
            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("OPENAI_API_KEY")

            # 最后回退到通用 Key（.env 中的 LLM_API_KEY）
            if not api_key:
                api_key = generic_key
            
            # Validate the API key
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export OPENAI_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['api_key'] = 'your-key-here'\n"
                    "3. Use the web interface to update the API key"
                )
            
            if api_key == "your-openai-api-key-here" or api_key == "":
                raise ValueError(
                    "Please replace the placeholder API key with your actual OpenAI API key. "
                    "You can get one from: https://platform.openai.com/api-keys"
                )
        elif provider == "anthropic":
            # First check if API key is provided in config
            api_key = self.config.get("anthropic_api_key")
            
            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                api_key = generic_key
            
            # Validate the API key
            if not api_key:
                raise ValueError(
                    "Anthropic API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export ANTHROPIC_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['anthropic_api_key'] = 'your-key-here'\n"
                )
            
            if api_key == "":
                raise ValueError(
                    "Please provide your actual Anthropic API key. "
                    "You can get one from: https://console.anthropic.com/"
                )
        elif provider == "qwen":
            # First check if API key is provided in config
            api_key = self.config.get("qwen_api_key")

            # If not in config, check environment variable
            if not api_key:
                api_key = os.environ.get("DASHSCOPE_API_KEY")

            if not api_key:
                api_key = generic_key

            # Validate the API key
            if not api_key:
                raise ValueError(
                    "Qwen API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export DASHSCOPE_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['qwen_api_key'] = 'your-key-here'\n"
                )

            if api_key == "":
                raise ValueError(
                    "Please provide your actual Qwen API key. "
                    "You can get one from: https://dashscope.console.aliyun.com/"
                )
        elif provider in MINIMAX_PROVIDER_CONFIG:
            provider_config = MINIMAX_PROVIDER_CONFIG[provider]
            api_key = self.config.get(provider_config["config_key"])

            if not api_key:
                for env_key in provider_config["env_keys"]:
                    api_key = os.environ.get(env_key)
                    if api_key:
                        break

            if not api_key:
                api_key = generic_key

            if not api_key:
                env_exports = "\n".join(
                    f"{idx}. Set environment variable: export {env_key}='your-key-here'"
                    for idx, env_key in enumerate(provider_config["env_keys"], start=1)
                )
                raise ValueError(
                    f"{provider_config['label']} API key not found. Please set it using one of these methods:\n"
                    f"{env_exports}\n"
                    f"{len(provider_config['env_keys']) + 1}. Update the config with: "
                    f"config['{provider_config['config_key']}'] = 'your-key-here'\n"
                    f"{len(provider_config['env_keys']) + 2}. Use the web interface to update the API key"
                )

            if api_key == "":
                raise ValueError(
                    f"Please provide your actual {provider_config['label']} API key. "
                    f"You can get one from: {provider_config['console_url']}"
                )
        elif provider == "gemini":
            api_key = self.config.get("gemini_api_key")

            if not api_key:
                api_key = os.environ.get("GOOGLE_API_KEY")

            if not api_key:
                api_key = generic_key

            if not api_key:
                raise ValueError(
                    "Google Gemini API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export GOOGLE_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['gemini_api_key'] = 'your-key-here'\n"
                    "3. Use the web interface to update the API key"
                )

            if api_key == "":
                raise ValueError(
                    "Please provide your actual Google Gemini API key. "
                    "You can get one from: https://ai.google.dev/"
                )
        elif provider == "deepseek":
            api_key = self.config.get("deepseek_api_key")

            if not api_key:
                api_key = os.environ.get("DEEPSEEK_API_KEY")

            if not api_key:
                api_key = generic_key

            if not api_key:
                raise ValueError(
                    "DeepSeek API key not found. Please set it using one of these methods:\n"
                    "1. Set environment variable: export DEEPSEEK_API_KEY='your-key-here'\n"
                    "2. Update the config with: config['deepseek_api_key'] = 'your-key-here'\n"
                    "3. Use the web interface to update the API key"
                )

            if api_key == "":
                raise ValueError(
                    "Please provide your actual DeepSeek API key. "
                    "You can get one from: https://platform.deepseek.com/api_keys"
                )
        elif provider == "bob-api":
            provider_config = BOB_API_PROVIDER_CONFIG
            api_key = self.config.get(provider_config["config_key"])
            if not api_key:
                api_key = os.environ.get(provider_config["env_key"])
            if not api_key:
                api_key = generic_key
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError(
                    "Bob API key not found. Please set BOB_API_KEY, configure "
                    "config['bob_api_key'], or use the web interface."
                )
            api_key = api_key.strip()
        else:
            raise ValueError(f"Unsupported provider: {provider}. Must be one of {', '.join(SUPPORTED_PROVIDERS)}")

        return api_key

    def _create_llm(
        self,
        provider: str,
        model: str,
        temperature: float,
        api_key: str = None,
        base_url: str = None,
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the provider.

        Args:
            provider: The provider name ("openai", "anthropic", "qwen", "minimax", "minimax_cn", "gemini" or "deepseek")
            model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022", "qwen-vl-max-latest", "MiniMax-M3")
            temperature: The temperature setting for the model
            api_key: 该槽位专属 Key；为空时回落到 _get_api_key(provider)
            base_url: 该槽位专属地址；为空时回落到通用 LLM_BASE_URL / provider 默认地址

        Returns:
            BaseChatModel: An instance of the appropriate LLM class
        """
        api_key = api_key or self._get_api_key(provider)
        # 槽位专属 base_url > 通用 LLM_BASE_URL
        base_url = base_url or self.config.get("llm_base_url") or None

        if provider == "openai":
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                **({"base_url": base_url} if base_url else {}),
            )
        elif provider == "anthropic":
            # ChatAnthropic handles SystemMessage extraction automatically
            # It extracts SystemMessage from the message list and passes it as 'system' parameter
            # The messages array should contain at least one non-SystemMessage
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        elif provider == "qwen":
            return ChatQwen(
                model=model,
                temperature=temperature,
                api_key=api_key,
                max_retries=4,
            )
        elif provider in MINIMAX_PROVIDER_CONFIG:
            # MiniMax uses OpenAI-compatible APIs; CN and global differ by base URL.
            # Temperature must be in (0.0, 1.0] for MiniMax.
            # MiniMax M-series models are reasoning models: without reasoning_split
            # they inline "<think>...</think>" into message content, which breaks
            # every downstream structured/JSON parser and can stall the agent graph.
            clamped_temp = max(0.01, min(temperature, 1.0))
            return ChatOpenAI(
                model=model,
                temperature=clamped_temp,
                api_key=api_key,
                openai_api_base=base_url or MINIMAX_PROVIDER_CONFIG[provider]["base_url"],
                extra_body={"reasoning_split": True},
                # ~110 tok/s measured; reasoning tokens also draw from this
                # budget. Measured: 8192 sometimes gets fully consumed by
                # thinking, leaving content empty (~40s wasted). 16384 keeps
                # room for both thinking and the report.
                max_tokens=16384,
                # A single request normally finishes in 20-90s. The server
                # occasionally stalls (runaway thinking), so cap at 120s and
                # leave retrying to the agent nodes (bounded retries); this
                # keeps a single logical call from blocking for 5+ minutes.
                timeout=120,
                max_retries=0,
            )
        elif provider == "gemini":
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                openai_api_base=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        elif provider == "deepseek":
            # DeepSeek provides an OpenAI-compatible text API.
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
                openai_api_base=base_url or "https://api.deepseek.com/v1",
            )
        elif provider == "bob-api":
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("Bob API key is required")
            if not isinstance(model, str) or not model.strip():
                raise ValueError("Bob API model name is required")
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
                or not 0 <= temperature <= 2
            ):
                raise ValueError("Bob API temperature must be a number between 0 and 2")
            return ChatOpenAI(
                model=model.strip(),
                temperature=temperature,
                api_key=api_key.strip(),
                openai_api_base=normalize_bob_api_base_url(base_url),
                timeout=BOB_API_PROVIDER_CONFIG["timeout"],
                max_retries=BOB_API_PROVIDER_CONFIG["max_retries"],
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Must be one of {', '.join(SUPPORTED_PROVIDERS)}")

    # def _set_tool_nodes(self) -> Dict[str, ToolNode]:
    #     """
    #     Define tool nodes for each agent type (indicator, pattern, trend).
    #     """
    #     return {
    #         "indicator": ToolNode(
    #             [
    #                 self.toolkit.compute_macd,
    #                 self.toolkit.compute_roc,
    #                 self.toolkit.compute_rsi,
    #                 self.toolkit.compute_stoch,
    #                 self.toolkit.compute_willr,
    #             ]
    #         ),
    #         "pattern": ToolNode(
    #             [
    #                 self.toolkit.generate_kline_image,
    #             ]
    #         ),
    #         "trend": ToolNode([self.toolkit.generate_trend_image]),
    #     }

    def refresh_llms(self):
        """
        Refresh the LLM objects with the current API key from environment.
        This is called when the API key is updated.
        """
        # Recreate LLM objects with current config values
        self.agent_llm = self._create_llm(
            provider=self.config.get("agent_llm_provider", "openai"),
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1),
            api_key=self.config.get("agent_llm_api_key"),
            base_url=self.config.get("agent_llm_base_url"),
        )
        self.graph_llm = self._create_llm(
            provider=self.config.get("graph_llm_provider", "openai"),
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1),
            api_key=self.config.get("graph_llm_api_key"),
            base_url=self.config.get("graph_llm_base_url"),
        )

        # Recreate the graph setup with new LLMs
        self.graph_setup = SetGraph(
            self.agent_llm,
            self.graph_llm,
            self.toolkit,
            # self.tool_nodes,
        )

        # Recreate the main graph
        self.graph = self.graph_setup.set_graph()

    def update_api_key(self, api_key: str, provider: str = "openai"):
        """
        Update the API key in the config and refresh LLMs.
        This method is called by the web interface when API key is updated.
        
        Args:
            api_key (str): The new API key
            provider (str): The provider name, defaults to "openai"
        """
        if provider == "openai":
            # Update the config with the new API key
            self.config["api_key"] = api_key
            
            # Also update the environment variable for consistency
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "anthropic":
            # Update the config with the new API key
            self.config["anthropic_api_key"] = api_key
            
            # Also update the environment variable for consistency
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "qwen":
            # Update the config with the new API key
            self.config["qwen_api_key"] = api_key

            # Also update the environment variable for consistency
            os.environ["DASHSCOPE_API_KEY"] = api_key
        elif provider == "minimax":
            # Update the config with the new API key
            self.config["minimax_api_key"] = api_key

            # Also update the environment variable for consistency
            os.environ["MINIMAX_API_KEY"] = api_key
        elif provider == "minimax_cn":
            # Update the config with the new API key
            self.config["minimax_cn_api_key"] = api_key

            # Keep CN credentials separate from the global MiniMax key.
            os.environ["MINIMAX_CN_API_KEY"] = api_key
        elif provider == "gemini":
            self.config["gemini_api_key"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
        elif provider == "deepseek":
            self.config["deepseek_api_key"] = api_key
            os.environ["DEEPSEEK_API_KEY"] = api_key
        elif provider == "bob-api":
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("Bob API key is required")
            api_key = api_key.strip()
            self.config["bob_api_key"] = api_key
            os.environ["BOB_API_KEY"] = api_key
        else:
            raise ValueError(f"Unsupported provider: {provider}. Must be one of {', '.join(SUPPORTED_PROVIDERS)}")

        # 同步通用 Key，保持与 .env 的 LLM_API_KEY 一致的读取路径
        self.config["llm_api_key"] = api_key

        # 同步槽位 Key：使用同一 provider 的槽位沿用新 Key，
        # 否则 .env 里的槽位 Key 会覆盖本次更新
        for slot in ("agent", "graph"):
            if self.config.get(f"{slot}_llm_provider") == provider:
                self.config[f"{slot}_llm_api_key"] = api_key

        # Refresh the LLMs with the new API key
        self.refresh_llms()
