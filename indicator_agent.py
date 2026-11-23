"""
Agent for technical indicator analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to compute and interpret indicators like MACD, RSI, ROC, Stochastic, and Williams %R.
"""

import copy
import json
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from graph_util import describe_llm


def message_text(content) -> str:
    """把 str / content-block 列表统一成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "\n".join(p for p in parts if p)
    return ""


def invoke_with_retry(call_fn, messages, label, attempts=2, wait_sec=3):
    """带重试的 LLM 调用；最终失败时抛出最后一次异常。"""
    last_error = None
    for attempt in range(attempts):
        try:
            return call_fn(messages)
        except Exception as exc:
            last_error = exc
            print(
                f"{label} attempt {attempt + 1}/{attempts} failed: "
                f"{type(exc).__name__}: {str(exc)[:120]}"
            )
            if attempt < attempts - 1:
                time.sleep(wait_sec)
    raise last_error


def normalize_tool_calls(ai_message):
    """把 tool_calls 与 invalid_tool_calls 合并成带唯一 id 的调用列表。

    OpenAI 兼容端（如 MiniMax）偶发返回缺失/重复的 call id，或 arguments
    不是合法 JSON。此时 langchain 会把它放进 invalid_tool_calls，而请求序列化
    仍会带上这些 tool_call；若没有对应的 ToolMessage 应答，下一轮就会报
    "tool call and result not match (2013)" 400。这里统一补齐 id 并标记非法项，
    保证每个 tool_call 都能被应答。
    """
    valid_calls = list(getattr(ai_message, "tool_calls", None) or [])
    invalid_calls = list(getattr(ai_message, "invalid_tool_calls", None) or [])
    valid_ids = {c.get("id") for c in valid_calls}

    calls = []
    used = set()
    for call in valid_calls + invalid_calls:
        call_id = call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
        while call_id in used:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
        used.add(call_id)
        args = call.get("args")
        calls.append(
            {
                "id": call_id,
                "name": call.get("name") or "",
                "args": args if isinstance(args, dict) else {},
                "invalid": call.get("id") not in valid_ids,
            }
        )
    return calls


def answer_tool_calls(calls, tools, state, messages):
    """执行调用并追加 ToolMessage，保证每个 call id 都有且仅有一条应答。"""
    for call in calls:
        if call["invalid"]:
            messages.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    content=json.dumps(
                        {
                            "error": "invalid tool call arguments, please call again with valid JSON args",
                            "tool": call["name"],
                        }
                    ),
                )
            )
            continue

        tool_args = dict(call["args"])
        tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
        tool_fn = next((t for t in tools if t.name == call["name"]), None)
        if tool_fn is None:
            messages.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    content=json.dumps({"error": f"unknown tool: {call['name']}"}),
                )
            )
            continue
        try:
            tool_result = tool_fn.invoke(tool_args)
        except Exception as exc:  # 工具异常也要应答，否则消息序列不完整
            messages.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    content=json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
                )
            )
            continue
        messages.append(
            ToolMessage(tool_call_id=call["id"], content=json.dumps(tool_result))
        )


def local_indicator_report(kline_data, tools):
    """不依赖 LLM 的确定性兜底报告，直接用本地工具计算结果。"""
    by_name = {t.name: t for t in tools}

    def tail(fn_name, **kwargs):
        fn = by_name.get(fn_name)
        if fn is None:
            return {}
        try:
            result = fn.invoke({"kline_data": kline_data, **kwargs})
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {
            key: (value[-1] if isinstance(value, list) and value else value)
            for key, value in result.items()
        }

    rsi = tail("compute_rsi", period=14).get("rsi")
    macd = tail("compute_macd")
    roc = tail("compute_roc", period=10).get("roc")
    stoch = tail("compute_stoch")
    willr = tail("compute_willr", period=14).get("willr")

    macd_hist = macd.get("macd_hist")
    if isinstance(macd_hist, (int, float)):
        bias = "bullish" if macd_hist > 0 else "bearish"
    else:
        bias = "neutral"

    return (
        "Indicator analysis (deterministic fallback, LLM unavailable):\n"
        f"- RSI(14): {rsi}\n"
        f"- MACD: {macd.get('macd')}, signal: {macd.get('macd_signal')}, "
        f"histogram: {macd_hist}\n"
        f"- ROC(10): {roc}\n"
        f"- Stochastic %K/%D: {stoch.get('stoch_k')}/{stoch.get('stoch_d')}\n"
        f"- Williams %R: {willr}\n"
        f"- Bias: {bias} (based on MACD histogram sign).\n"
    )


def create_indicator_agent(llm, toolkit):
    """
    Create an indicator analysis agent node for HFT. The agent uses LLM and indicator tools to analyze OHLCV data.
    """

    def indicator_agent_node(state):
        # --- Tool definitions ---
        tools = [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ]
        time_frame = state["time_frame"]
        # --- System prompt for LLM ---
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a high-frequency trading (HFT) analyst assistant operating under time-sensitive conditions. "
                    "You must analyze technical indicators to support fast-paced trading execution.\n\n"
                    "You have access to tools: compute_rsi, compute_macd, compute_roc, compute_stoch, and compute_willr. "
                    "Use them by providing appropriate arguments like `kline_data` and the respective periods.\n\n"
                    f"⚠️ The OHLC data provided is from a {time_frame} intervals, reflecting recent market behavior. "
                    "You must interpret this data quickly and accurately.\n\n"
                    "Here is the OHLC data:\n{kline_data}.\n\n"
                    "Call necessary tools, and analyze the results.\n",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(kline_data=json.dumps(state["kline_data"], indent=2))

        tool_chain = prompt | llm.bind_tools(tools)
        # 第二轮起不再绑定工具：推理模型在有工具上下文时会持续“规划”，
        # 实测会把 max_tokens 全部耗在思考上（正文为空，甚至请求挂到超时）。
        text_chain = prompt | llm

        messages = list(state.get("messages", []) or [])
        if not messages:
            messages = [HumanMessage(content="Begin indicator analysis.")]

        final_text = ""
        used_llm = False

        # --- Step 1: 第一轮允许调用指标工具 ---
        try:
            ai_response = invoke_with_retry(
                tool_chain.invoke, messages, "indicator tool round"
            )
        except Exception:
            ai_response = None

        if ai_response is not None:
            calls = normalize_tool_calls(ai_response)
            text = message_text(ai_response.content).strip()
            # 有工具调用的那一轮，content 往往只是“我这就去算…”之类的开场白，
            # 不能当作最终报告；必须等第二轮纯文本收敛。
            if text and not calls:
                final_text = text
                used_llm = True

            extra_kwargs = {
                k: v
                for k, v in (ai_response.additional_kwargs or {}).items()
                if k != "tool_calls"
            }
            messages.append(
                AIMessage(
                    content=ai_response.content or "",
                    additional_kwargs=extra_kwargs,
                    tool_calls=[
                        {
                            "id": c["id"],
                            "name": c["name"],
                            "args": c["args"],
                            "type": "tool_call",
                        }
                        for c in calls
                    ],
                )
            )
            if calls:
                answer_tool_calls(calls, tools, state, messages)

        # --- Step 2: 纯文本收敛成最终报告 ---
        if not final_text:
            if messages and isinstance(messages[-1], ToolMessage):
                messages.append(
                    HumanMessage(
                        content=(
                            "Do not call any more tools. Using the indicator values above, "
                            "write the final indicator report in plain text (<=250 words): "
                            "trend/momentum direction, MACD, RSI, ROC, Stochastic, Williams %R "
                            "readings, and the resulting bullish/bearish bias."
                        )
                    )
                )
            try:
                summary = invoke_with_retry(
                    text_chain.invoke, messages, "indicator summary"
                )
                final_text = message_text(summary.content).strip()
                if final_text:
                    messages.append(AIMessage(content=final_text))
                    used_llm = True
            except Exception:
                final_text = ""

        # --- Step 3: LLM 全部失败时用本地指标计算结果兜底 ---
        if not final_text:
            final_text = local_indicator_report(state["kline_data"], tools)

        model_name, model_provider = describe_llm(llm)
        provenance = {
            "indicator": {
                "method": "llm" if used_llm else "local_fallback",
                "model": model_name if used_llm else None,
                "provider": model_provider if used_llm else None,
                "detail": (
                    f"大模型 {model_name} 调用 compute_* 工具后归纳生成"
                    if used_llm
                    else f"未调用大模型（{model_name} 调用失败），改用本地工具确定性计算"
                ),
            }
        }

        return {
            "messages": messages,
            "indicator_report": final_text,
            "analysis_provenance": provenance,
        }

    return indicator_agent_node
