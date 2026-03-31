"""
Tool-calling helpers (same contract as 1min-relay txt_func, without Flask).
Used only by openclaw-bridge; 1min-relay no longer injects tool prompts for OpenClaw.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple


def build_tool_calling_instructions(tools: list) -> str:
    fn_tools = []
    for t in tools or []:
        if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn_tools.append(t)
    if not fn_tools:
        return ""
    try:
        tools_json = json.dumps(fn_tools, ensure_ascii=False)
    except Exception:
        tools_json = "[]"

    return (
        "TOOL CALLING MODE (OpenAI-compatible emulation)\n"
        "You MAY call tools. If the user asks to read/write files or update memory, you SHOULD use the appropriate tool.\n"
        "If you decide to call a tool, respond with ONLY a JSON object and nothing else.\n"
        "Schema:\n"
        "{\n"
        '  "tool_calls": [\n'
        "    {\n"
        '      "id": "call_1",\n'
        '      "type": "function",\n'
        '      "function": {\n'
        '        "name": "<tool name>",\n'
        '        "arguments": { <json object arguments> }\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "If you do NOT need a tool, respond normally with plain text.\n"
        "Available tools (OpenAI tools JSON):\n"
        f"{tools_json}\n"
    )


def maybe_extract_tool_calls_from_text(text: str) -> Tuple[str, Optional[List[dict]]]:
    if not isinstance(text, str):
        return "", None
    raw = text.strip()
    if not raw:
        return "", None

    try:
        lines = [ln.strip() for ln in raw.splitlines()]
        tokens = [ln for ln in lines if ln]
        if len(tokens) >= 3 and tokens[0].lower() == "tool":
            tool_name = tokens[1]
            args_raw = tokens[2]
            if args_raw.startswith("{") and args_raw.endswith("}"):
                try:
                    args_obj = json.loads(args_raw)
                except Exception:
                    args_obj = None
                if isinstance(args_obj, dict) and tool_name:
                    return (
                        "",
                        [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": str(tool_name),
                                    "arguments": json.dumps(args_obj, ensure_ascii=False),
                                },
                            }
                        ],
                    )
    except Exception:
        pass

    raw2 = raw
    if raw2.startswith("```"):
        raw2 = raw2.strip("`").strip()
        if raw2.lower().startswith("json"):
            raw2 = raw2[4:].strip()
        else:
            raw2 = raw2

    if not (raw2.startswith("{") and raw2.endswith("}")):
        return text, None

    try:
        obj = json.loads(raw2)
    except Exception:
        return text, None

    tool_calls = obj.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return text, None

    normalized = []
    for i, call in enumerate(tool_calls, 1):
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name") or call.get("name")
        args = fn.get("arguments")
        if not name:
            continue
        if isinstance(args, (dict, list)):
            args_str = json.dumps(args, ensure_ascii=False)
        elif args is None:
            args_str = "{}"
        else:
            args_str = str(args)
        normalized.append(
            {
                "id": call.get("id") or f"call_{i}",
                "type": "function",
                "function": {"name": str(name), "arguments": args_str},
            }
        )

    if not normalized:
        return text, None

    return "", normalized


def has_function_tools(tools: Any) -> bool:
    if not isinstance(tools, list):
        return False
    return any(isinstance(t, dict) and t.get("type") == "function" for t in tools)


def augment_messages_with_tools(messages: list, tools: list) -> list:
    instr = build_tool_calling_instructions(tools)
    if not instr:
        return list(messages)
    msgs = [dict(m) for m in messages if isinstance(m, dict)]
    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str):
                msgs[i] = {**m, "content": f"{instr}\n\n{c}"}
            else:
                msgs[i] = {**m, "content": instr}
            return msgs
    return [{"role": "system", "content": instr}, *msgs]
