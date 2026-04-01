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

    # We must include the *actual* parameter names/required fields (e.g. pathAlias vs path),
    # otherwise OpenClaw tool execution may fail and cause many retries.
    # To keep prompts small we include a minimized tool schema (name/description/parameters only).
    minimized = []
    for t in fn_tools:
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        minimized.append(
            {
                "type": "function",
                "function": {
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": params,
                },
            }
        )
    try:
        tools_hint = json.dumps(minimized, ensure_ascii=False)
    except Exception:
        tools_hint = "[]"

    return (
        "TOOL CALLING MODE (OpenAI-compatible emulation)\n"
        "\n"
        "You MAY call tools.\n"
        "\n"
        "STRICT TOOLS CONTRACT (IMPORTANT):\n"
        "- If you claim you read/write/update/create/delete/execute/send anything outside the chat, you MUST return tool_calls.\n"
        "- Do NOT say you changed files / sent messages / executed commands unless tool_calls are present.\n"
        "- When you return tool_calls, return ONLY a single JSON object (no prose, no markdown, no code fences).\n"
        "- tool_calls MUST be a non-empty array when you choose to call tools.\n"
        "- Arguments MUST follow the provided JSON Schema exactly (including required fields and exact key names).\n"
        "\n"
        "General guidance:\n"
        "- If you need exact oldText matching for an edit/replace tool: first call the read tool and copy the exact substring.\n"
        "- For small files, prefer writing the full new content to avoid fragile edit matches.\n"
        "\n"
        "Output schema (example):\n"
        "{\n"
        '  \"tool_calls\": [\n'
        "    {\n"
        '      \"id\": \"call_1\",\n'
        '      \"type\": \"function\",\n'
        '      \"function\": {\n'
        '        \"name\": \"<tool name>\",\n'
        '        \"arguments\": { \"key\": \"value\" }\n'
        "      }\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "If you do NOT need a tool, respond normally with plain text.\n"
        "\n"
        "Available tools (minimized OpenAI tools JSON):\n"
        f"{tools_hint}\n"
    )


def maybe_extract_tool_calls_from_text(text: str) -> Tuple[str, Optional[List[dict]]]:
    if not isinstance(text, str):
        return "", None
    raw = text.strip()
    if not raw:
        return "", None

    # OpenClaw-style shorthand: function_name(key="value", n=1, flag=true)
    # Example: memory_search(query="Память", maxResults=3, minScore=0.0)
    try:
        import re

        m = re.match(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", raw)
        if m:
            fname = m.group(1)
            args_src = m.group(2).strip()
            args_obj: dict[str, Any] = {}
            if args_src:
                # split by commas at top-level (no nesting support; best-effort)
                parts = [p.strip() for p in args_src.split(",") if p.strip()]
                for p in parts:
                    if "=" not in p:
                        continue
                    k, v = p.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # strip quotes
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v2 = v[1:-1]
                        args_obj[k] = v2
                        continue
                    # bool/null
                    vl = v.lower()
                    if vl in ("true", "false"):
                        args_obj[k] = (vl == "true")
                        continue
                    if vl in ("null", "none"):
                        args_obj[k] = None
                        continue
                    # int/float
                    try:
                        if "." in v or "e" in vl:
                            args_obj[k] = float(v)
                        else:
                            args_obj[k] = int(v)
                        continue
                    except Exception:
                        args_obj[k] = v
            if fname:
                return (
                    "",
                    [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": str(fname), "arguments": json.dumps(args_obj, ensure_ascii=False)},
                        }
                    ],
                )
    except Exception:
        pass

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
