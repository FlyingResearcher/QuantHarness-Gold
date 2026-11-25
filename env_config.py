"""LLM 配置加载器：配置唯一来源是 `.env` 文件 / 系统环境变量。

不再使用 default_config.py 作为配置项，Python 代码中也不保存任何 API Key。

优先级：系统环境变量 > 项目根目录 `.env`。
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

BOB_API_DEFAULT_BASE_URL = "https://api.bob-api.com/"
BOB_API_DEFAULT_MODEL = "deepseek-v4.1-flash"

# provider -> 配置字典中对应的 key 字段（沿用既有 provider 分支逻辑）
_PROVIDER_KEY_FIELDS = {
    "openai": "api_key",
    "anthropic": "anthropic_api_key",
    "qwen": "qwen_api_key",
    "minimax": "minimax_api_key",
    "minimax_cn": "minimax_cn_api_key",
    "gemini": "gemini_api_key",
    "deepseek": "deepseek_api_key",
    "bob-api": "bob_api_key",
}


def load_env_file(path: Path = ENV_FILE) -> dict:
    """解析 .env 文件（支持注释、空行、单/双引号包裹的值）。

    已有系统环境变量优先，不会被文件覆盖。
    """
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        values[key] = value
        os.environ.setdefault(key, value)
    return values


_FILE_ENV = load_env_file()


def get_env(key: str, default=None):
    """读取配置：系统环境变量优先，其次 .env，最后默认值。"""
    value = os.environ.get(key)
    if value not in (None, ""):
        return value
    value = _FILE_ENV.get(key)
    if value not in (None, ""):
        return value
    return default


def _float_env(key: str, default: float) -> float:
    try:
        return float(get_env(key, str(default)))
    except ValueError:
        return default


def build_config() -> dict:
    """从环境变量构建 TradingGraph 所需配置字典。

    支持「双槽位」多模型配置：
      - agent 槽位：分析（工具调用 / 文本）模型
      - graph 槽位：视觉识别 / 最终决策模型

    两个槽位都可以用 LLM_AGENT_* / LLM_GRAPH_* 单独覆盖
    provider / base_url / model / api_key / temperature；
    未配置时回落到通用的 LLM_*。
    """
    provider = get_env("LLM_PROVIDER", "openai")
    model = get_env("LLM_MODEL")
    api_key = get_env("LLM_API_KEY")
    base_url = get_env("LLM_BASE_URL")
    try:
        temperature = float(get_env("LLM_TEMPERATURE", "0.1"))
    except ValueError:
        temperature = 0.1

    config = {
        # 通用（兜底）配置
        "llm_base_url": base_url,
        "llm_api_key": api_key,
        # agent 槽位
        "agent_llm_provider": get_env("LLM_AGENT_PROVIDER", provider),
        "agent_llm_model": get_env("LLM_AGENT_MODEL", model),
        "agent_llm_base_url": get_env("LLM_AGENT_BASE_URL", base_url),
        "agent_llm_api_key": get_env("LLM_AGENT_API_KEY", api_key),
        "agent_llm_temperature": _float_env("LLM_AGENT_TEMPERATURE", temperature),
        # graph 槽位
        "graph_llm_provider": get_env("LLM_GRAPH_PROVIDER", provider),
        "graph_llm_model": get_env("LLM_GRAPH_MODEL", model),
        "graph_llm_base_url": get_env("LLM_GRAPH_BASE_URL", base_url),
        "graph_llm_api_key": get_env("LLM_GRAPH_API_KEY", api_key),
        "graph_llm_temperature": _float_env("LLM_GRAPH_TEMPERATURE", temperature),
    }

    # Bob API 支持独立的密钥、地址和模型环境变量，同时保留双槽位覆盖能力。
    for slot in ("agent", "graph"):
        if config[f"{slot}_llm_provider"] == "bob-api":
            config[f"{slot}_llm_model"] = config[f"{slot}_llm_model"] or get_env(
                "BOB_API_MODEL", BOB_API_DEFAULT_MODEL
            )
            config[f"{slot}_llm_base_url"] = config[f"{slot}_llm_base_url"] or get_env(
                "BOB_API_BASE_URL", BOB_API_DEFAULT_BASE_URL
            )
            config[f"{slot}_llm_api_key"] = config[f"{slot}_llm_api_key"] or get_env("BOB_API_KEY")

    missing_models = [
        slot
        for slot in ("agent", "graph")
        if not config[f"{slot}_llm_model"]
    ]
    if missing_models:
        raise RuntimeError(
            "缺少模型名称：请配置 LLM_MODEL，"
            "或分别配置 LLM_AGENT_MODEL / LLM_GRAPH_MODEL"
        )

    missing_keys = [
        slot
        for slot in ("agent", "graph")
        if not config[f"{slot}_llm_api_key"]
    ]
    if missing_keys:
        raise RuntimeError(
            "缺少 API Key：请配置 LLM_API_KEY，"
            "或分别配置 LLM_AGENT_API_KEY / LLM_GRAPH_API_KEY"
        )

    # 各 provider 的 Key 字段统一由通用 Key 填充，未使用的置空（兼容 Web 界面校验）
    for field in _PROVIDER_KEY_FIELDS.values():
        config.setdefault(field, "")
    key_field = _PROVIDER_KEY_FIELDS.get(provider)
    if key_field:
        config[key_field] = api_key

    bob_slot = next(
        (slot for slot in ("agent", "graph") if config[f"{slot}_llm_provider"] == "bob-api"),
        None,
    )
    config["bob_api_key"] = config["bob_api_key"] or (
        config[f"{bob_slot}_llm_api_key"] if bob_slot else get_env("BOB_API_KEY", "")
    )
    config["bob_api_base_url"] = (
        config[f"{bob_slot}_llm_base_url"]
        if bob_slot
        else get_env("BOB_API_BASE_URL", BOB_API_DEFAULT_BASE_URL)
    )
    config["bob_api_model"] = (
        config[f"{bob_slot}_llm_model"]
        if bob_slot
        else get_env("BOB_API_MODEL", BOB_API_DEFAULT_MODEL)
    )

    return config
