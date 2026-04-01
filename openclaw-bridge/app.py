"""
OpenClaw tool-calling bridge: OpenAI-compatible /v1/chat/completions in front of 1min-relay.
Strips tools from upstream, injects synthetic tool instructions, parses JSON tool_calls from model text.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict
import hashlib

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tool_parse import augment_messages_with_tools, has_function_tools, maybe_extract_tool_calls_from_text

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "https://api.ratu.sh").rstrip("/")
BRIDGE_SECRET = os.environ.get("BRIDGE_SECRET", "").strip()
REQUEST_TIMEOUT = float(os.environ.get("BRIDGE_UPSTREAM_TIMEOUT", "120"))
LOOP_WINDOW_S = int(os.environ.get("BRIDGE_LOOP_WINDOW_S", "180"))
LOOP_MAX_CALLS = int(os.environ.get("BRIDGE_LOOP_MAX_CALLS", "8"))
TOOL_FALLBACK_MODEL = os.environ.get("BRIDGE_TOOL_FALLBACK_MODEL", "").strip()
FORCE_TOOL_MODEL = os.environ.get("BRIDGE_FORCE_TOOL_MODEL", "").strip()
STRICT_TOOL_CONTRACT = os.environ.get("BRIDGE_STRICT_TOOL_CONTRACT", "1").strip().lower() not in ("0", "false", "no", "off", "")

app = FastAPI(title="openclaw-bridge", version="1.0.0")

_loop_guard: Dict[str, list[float]] = {}

def _is_effectively_empty(s: Any) -> bool:
    if not isinstance(s, str):
        return True
    t = s.replace("\u200b", "").replace("\ufeff", "").replace("\u2060", "")
    return not t.strip()


async def _collect_openai_sse_text(resp: httpx.Response) -> str:
    """
    Collect assistant text from OpenAI-style SSE stream:
      data: {"choices":[{"delta":{"content":"..."}}]}
      data: [DONE]
    Best-effort; ignores any other fields.
    """
    collected: list[str] = []
    buffer = ""
    async for chunk in resp.aiter_text():
        if not chunk:
            continue
        buffer += chunk
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            line = block.strip()
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            choices = obj.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            c0 = choices[0] if isinstance(choices[0], dict) else {}
            delta = c0.get("delta") if isinstance(c0.get("delta"), dict) else {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                collected.append(piece)
    return "".join(collected)

def _tool_required_map(tools: Any) -> Dict[str, list]:
    req: Dict[str, list] = {}
    if not isinstance(tools, list):
        return req
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        name = fn.get("name")
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        required = params.get("required") if isinstance(params.get("required"), list) else []
        if isinstance(name, str) and name:
            req[name] = [str(x) for x in required if isinstance(x, (str, int, float, bool))]
    return req


def _tool_calls_missing_required(tool_calls: Any, required_map: Dict[str, list]) -> str | None:
    if not isinstance(tool_calls, list) or not required_map:
        return None
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        required = required_map.get(name) or []
        if not required:
            continue
        args_raw = fn.get("arguments")
        args_obj: Any = None
        if isinstance(args_raw, str):
            try:
                args_obj = json.loads(args_raw)
            except Exception:
                args_obj = None
        elif isinstance(args_raw, dict):
            args_obj = args_raw
        else:
            args_obj = None
        args_obj = args_obj if isinstance(args_obj, dict) else {}
        missing = [k for k in required if k not in args_obj]
        if missing:
            return f"tool '{name}' missing required keys: {missing}"
    return None


def _upstream_headers(request: Request) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("Authorization") or ""
    if auth:
        headers["Authorization"] = auth
    else:
        ak = request.headers.get("API-KEY") or request.headers.get("Api-Key") or ""
        if ak:
            headers["API-KEY"] = ak
    return headers

def _loop_key(body: Dict[str, Any], tools: Any) -> str:
    """
    Build a stable fingerprint for the likely "same request" loop.
    We use: model + last user content (first 2KB) + tool names + tool_count.
    """
    model = str(body.get("model") or "").strip()
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    last_user = ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                last_user = c
            elif isinstance(c, list):
                parts = []
                for it in c:
                    if isinstance(it, dict) and isinstance(it.get("text"), str):
                        parts.append(it["text"])
                last_user = "\n".join(parts)
            break
    last_user = (last_user or "")[:2048]
    names = []
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function":
                fn = t.get("function") if isinstance(t.get("function"), dict) else {}
                n = fn.get("name")
                if isinstance(n, str):
                    names.append(n)
    names = sorted(set(names))[:200]
    raw = json.dumps({"m": model, "u": last_user, "tn": names, "tc": len(names)}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _loop_guard_allow(key: str) -> bool:
    now = time.time()
    window_start = now - LOOP_WINDOW_S
    times = _loop_guard.get(key) or []
    times = [t for t in times if t >= window_start]
    times.append(now)
    _loop_guard[key] = times
    return len(times) <= LOOP_MAX_CALLS


def _last_user_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts: list[str] = []
            for it in c:
                if isinstance(it, dict) and isinstance(it.get("text"), str):
                    parts.append(it["text"])
            return "\n".join(parts)
        return ""
    return ""


def _looks_like_side_effect_claim(text: str) -> bool:
    """
    Detect "I already did X" claims that MUST be backed by tool_calls.
    Language-agnostic-ish: a small set of RU/EN stems for edits/reads/writes/exec/sends.
    This is not per-tool; it's a general truthfulness gate to prevent hallucinated writes.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    # common file/tool claims RU (use stems to cover gender/tense)
    ru = (
        "обновл",   # обновил/обновила/обновлено/обновлена/обновлены
        "запис",    # записал/записала/записано/записана
        "измен",    # изменил/изменила/изменено/изменена
        "добавл",   # добавил/добавила/добавлено/добавлена
        "удал",     # удалил/удалила/удалено/удалена
        "созда",    # создал/создала/создано/создана
        "прочит",   # прочитал/прочитала/прочитано/прочитана
        "отправ",   # отправил/отправила/отправлено/отправлена
        "выполн",   # выполнил/выполнила/выполнено/выполнена
        "запуст",   # запустил/запустила/запущено/запущена
        "перезапуст",  # перезапустил/перезапустила
    )
    # common file/tool claims EN
    en = (
        "i updated",
        "updated ",
        "i wrote",
        "wrote ",
        "i changed",
        "changed ",
        "i edited",
        "edited ",
        "i created",
        "created ",
        "i deleted",
        "deleted ",
        "i removed",
        "removed ",
        "i read",
        "here is the file",
        "here's the file",
        "i ran",
        "i executed",
        "i sent",
    )
    if any(k in t for k in ru) or any(k in t for k in en):
        return True
    # explicit filename mentions often indicate claimed file interaction
    if ".md" in t or "memory.md" in t:
        if any(k in t for k in ("обнов", "проч", "write", "read", "edit", "update")):
            return True
    return False


def _looks_like_external_read_request(user_text: str) -> bool:
    """
    Detect user requests that *require* reading external state (files, logs, configs).
    This is not per-tool; it's a general "show me / quote / contents" detector.
    """
    t = (user_text or "").strip().lower()
    if not t:
        return False
    wants_show = any(
        k in t
        for k in (
            "прочитай",
            "прочти",
            "покажи",
            "пришли",
            "выведи",
            "процит",
            "цитир",
            "содержим",
            "что в файле",
            "show me",
            "print",
            "output",
            "quote",
            "paste",
            "contents",
            "read the file",
        )
    )
    mentions_artifact = any(k in t for k in (".md", ".json", ".txt", "memory.md", "файл", "лог", "конфиг", "config"))
    return wants_show and mentions_artifact


def _looks_like_external_update_request(user_text: str) -> bool:
    """
    Detect user requests that require changing external state (files/configs/etc).
    Generic (not per-tool): update/edit/write/create/delete + mentions an artifact.
    """
    t = (user_text or "").strip().lower()
    if not t:
        return False
    wants_change = any(
        k in t
        for k in (
            "обнов",
            "измени",
            "внеси",
            "добав",
            "запиш",
            "удал",
            "созда",
            "update",
            "edit",
            "change",
            "write",
            "append",
            "delete",
            "create",
        )
    )
    mentions_artifact = any(k in t for k in (".md", ".json", ".txt", "memory.md", "файл", "лог", "конфиг", "config"))
    return wants_change and mentions_artifact


def _sse_chunks(content: str, model: str, prompt_tokens: int):
    words = content.split()
    chunks = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]
    if not chunks and content:
        chunks = [content]
    cid = f"chatcmpl-{uuid.uuid4()}"
    for chunk in chunks:
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    final = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": max(1, len(content) // 4),
            "total_tokens": prompt_tokens + max(1, len(content) // 4),
        },
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

def _sse_tool_calls(tool_calls: list, model: str, prompt_tokens: int):
    """
    Best-effort OpenAI-style SSE for tool_calls.
    Some clients (including gateways) request stream:true and may hang if they receive a plain JSON response.
    We emit a single chunk that contains tool_calls and then a final finish_reason=tool_calls + [DONE].
    """
    cid = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())
    # single payload chunk
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    final = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 1,
            "total_tokens": prompt_tokens + 1,
        },
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {"status": "ok", "upstream": UPSTREAM_BASE_URL}


@app.get("/v1/models")
async def models(request: Request):
    url = f"{UPSTREAM_BASE_URL}/v1/models"
    headers = _upstream_headers(request)
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, headers=headers)
        try:
            return JSONResponse(r.json(), status_code=r.status_code)
        except Exception:
            return JSONResponse({"error": r.text}, status_code=r.status_code)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if BRIDGE_SECRET:
        if (request.headers.get("X-Bridge-Secret") or "").strip() != BRIDGE_SECRET:
            raise HTTPException(status_code=401, detail="Invalid X-Bridge-Secret")

    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    headers = _upstream_headers(request)
    tools = body.get("tools") or []
    stream = bool(body.get("stream", False))
    model = (body.get("model") or "gpt-4o-mini").strip()
    required_map = _tool_required_map(tools)
    lk = _loop_key(body, tools)
    if has_function_tools(tools) and not _loop_guard_allow(lk):
        # Stop token-draining loops: OpenClaw may interpret non-2xx as API rate limit and enter cooldown.
        # Return a normal assistant response (200) instead.
        final_text = (
            "Loop guard: слишком много повторов одного и того же tool-запроса. "
            "Остановлено, чтобы не сжигать токены. "
            "Сделайте /reset и повторите запрос; если проблема повторится — это означает, что tool_calls "
            "ломаются (алиасы/аргументы edit/write) и их нужно исправлять."
        )
        if stream:
            def gen_lg():
                yield from _sse_chunks(final_text, model or "unknown", 1)
            return StreamingResponse(gen_lg(), media_type="text/event-stream")
        out = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or "unknown",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": final_text}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": max(1, len(final_text) // 4), "total_tokens": 1 + max(1, len(final_text) // 4)},
        }
        return JSONResponse(out)

    url = f"{UPSTREAM_BASE_URL}/v1/chat/completions"

    if not has_function_tools(tools):
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            if stream:

                async def proxy_stream() -> AsyncIterator[bytes]:
                    async with client.stream("POST", url, json=body, headers=headers) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk

                return StreamingResponse(proxy_stream(), media_type="text/event-stream")

            resp = await client.post(url, json=body, headers=headers)
            try:
                return JSONResponse(resp.json(), status_code=resp.status_code)
            except Exception:
                return JSONResponse({"error": resp.text}, status_code=resp.status_code)

    inner = {k: v for k, v in body.items() if k not in ("tools", "tool_choice", "parallel_tool_calls")}
    inner["stream"] = False
    inner["messages"] = augment_messages_with_tools(body.get("messages") or [], tools)
    # When tools are present, optionally force a more reliable model for tool-calling.
    if FORCE_TOOL_MODEL:
        inner["model"] = FORCE_TOOL_MODEL
        model = FORCE_TOOL_MODEL

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(url, json=inner, headers=headers)

    if resp.status_code != 200:
        try:
            return JSONResponse(resp.json(), status_code=resp.status_code)
        except Exception:
            return JSONResponse({"error": resp.text}, status_code=resp.status_code)

    data = resp.json()
    choice0 = (data.get("choices") or [{}])[0] or {}
    msg = (choice0.get("message") or {}) if isinstance(choice0.get("message"), dict) else {}
    content = msg.get("content")
    if not isinstance(content, str):
        content = ""
    clean, tool_calls = maybe_extract_tool_calls_from_text(content)

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    pt = int(usage.get("prompt_tokens") or 0)

    # Universal strict tool contract:
    # If tools were provided but the model returned plain text without tool_calls, we do one strict retry when:
    # - the assistant looks like it claims side-effects without tools, OR
    # - the user request clearly requires reading external state (files/logs/configs) as evidence.
    user_text = _last_user_text(inner.get("messages"))
    needs_read = _looks_like_external_read_request(user_text)
    needs_update = _looks_like_external_update_request(user_text)
    bad_claim = _looks_like_side_effect_claim(clean or content)
    strict_triggered = False
    if STRICT_TOOL_CONTRACT and has_function_tools(tools) and not tool_calls and (bad_claim or needs_read or needs_update):
        strict_triggered = True
        try:
            forced = dict(inner)
            if TOOL_FALLBACK_MODEL:
                forced["model"] = TOOL_FALLBACK_MODEL
            directive = (
                "TOOLS CONTRACT (STRICT): If tools are provided, you MUST NOT claim any external action "
                "(read/write/update/create/delete/execute/send) unless you return tool_calls that perform it. "
                "Return ONLY a JSON object with tool_calls. tool_calls MUST be a non-empty array."
            )
            forced["messages"] = list(inner.get("messages") or []) + [
                {
                    "role": "system",
                    "content": directive,
                }
            ]
            r_force = await client.post(url, json=forced, headers=headers)
            if r_force.status_code == 200:
                d_force = r_force.json()
                c_force = (d_force.get("choices") or [{}])[0] or {}
                m_force = (c_force.get("message") or {}) if isinstance(c_force.get("message"), dict) else {}
                t_force = m_force.get("content") if isinstance(m_force.get("content"), str) else ""
                clean_f, tool_calls_f = maybe_extract_tool_calls_from_text(t_force)
                if tool_calls_f and not _tool_calls_missing_required(tool_calls_f, required_map):
                    tool_calls = tool_calls_f
                    clean = clean_f
                    data = d_force
                    usage = d_force.get("usage") if isinstance(d_force.get("usage"), dict) else usage
                    pt = int((usage or {}).get("prompt_tokens") or pt)
                    # reflect model switch in our response model field for transparency
                    if TOOL_FALLBACK_MODEL:
                        model = TOOL_FALLBACK_MODEL
        except Exception:
            pass

    # If strict contract was triggered but we still have no tool_calls, return a hard failure message
    # instead of misleading "I updated/read/executed" prose.
    if strict_triggered and not tool_calls:
        final_text = (
            "Ошибка tool-calling: модель не вернула корректные tool_calls при запросе, который требует "
            "чтения/изменения внешнего состояния. Действие НЕ выполнено. "
            "Попробуйте повторить запрос или временно переключить модель на более надёжную для tool-calling."
        )
        if stream:
            def gen_tc_fail():
                yield from _sse_chunks(final_text, model, pt or 1)
            return StreamingResponse(gen_tc_fail(), media_type="text/event-stream")
        ct = max(1, len(final_text) // 4)
        return JSONResponse(
            {
                "id": data.get("id") or f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": data.get("created") or int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": final_text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt or 1, "completion_tokens": ct, "total_tokens": (pt or 1) + ct},
            }
        )

    if tool_calls:
        # Validate required parameters against the provided tools schema.
        # If invalid, do ONE retry asking for corrected tool_calls JSON only.
        miss = _tool_calls_missing_required(tool_calls, required_map)
        if miss:
            try:
                inner_retry = dict(inner)
                inner_retry["messages"] = list(inner.get("messages") or []) + [
                    {
                        "role": "system",
                        "content": (
                            "Your previous tool_calls JSON is invalid for the provided tool schema: "
                            f"{miss}. Return ONLY a corrected JSON object with tool_calls."
                        ),
                    }
                ]
                r2 = await client.post(url, json=inner_retry, headers=headers)
                if r2.status_code == 200:
                    d2 = r2.json()
                    c2 = (d2.get("choices") or [{}])[0] or {}
                    m2 = (c2.get("message") or {}) if isinstance(c2.get("message"), dict) else {}
                    t2 = m2.get("content") if isinstance(m2.get("content"), str) else ""
                    clean2, tool_calls2 = maybe_extract_tool_calls_from_text(t2)
                    if tool_calls2 and not _tool_calls_missing_required(tool_calls2, required_map):
                        tool_calls = tool_calls2
                        clean = clean2
                        data = d2
                        usage = d2.get("usage") if isinstance(d2.get("usage"), dict) else usage
                        pt = int((usage or {}).get("prompt_tokens") or pt)
            except Exception:
                pass

        if stream:
            def gen_tc():
                yield from _sse_tool_calls(tool_calls, model, pt or 1)
            return StreamingResponse(gen_tc(), media_type="text/event-stream")

        out = {
            "id": data.get("id") or f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": data.get("created") or int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": usage,
        }
        return JSONResponse(out)

    # Fallback: some upstreams return empty content for non-stream tool prompts.
    # Retry once using upstream streaming and collect full text.
    if _is_effectively_empty(clean if clean else content):
        try:
            inner2 = dict(inner)
            inner2["stream"] = True
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                async with client.stream("POST", url, json=inner2, headers=headers) as sresp:
                    if sresp.status_code == 200:
                        streamed_text = await _collect_openai_sse_text(sresp)
                        if isinstance(streamed_text, str) and streamed_text:
                            clean2, tool_calls2 = maybe_extract_tool_calls_from_text(streamed_text)
                            if tool_calls2:
                                if stream:
                                    def gen_tc2():
                                        yield from _sse_tool_calls(tool_calls2, model, pt or 1)
                                    return StreamingResponse(gen_tc2(), media_type="text/event-stream")

                                out = {
                                    "id": data.get("id") or f"chatcmpl-{uuid.uuid4()}",
                                    "object": "chat.completion",
                                    "created": data.get("created") or int(time.time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": {"role": "assistant", "content": "", "tool_calls": tool_calls2},
                                            "finish_reason": "tool_calls",
                                        }
                                    ],
                                    "usage": usage,
                                }
                                return JSONResponse(out)
                            # use streamed text as final assistant content
                            clean = clean2 if isinstance(clean2, str) else ""
                            content = streamed_text
        except Exception:
            pass

    final_text = clean if clean else content
    if not (final_text or "").strip():
        final_text = (
            "Ошибка: upstream вернул пустой ответ после запроса с tools. "
            "Повторите запрос или смените модель; убедитесь, что 1min-relay доступен."
        )

    if stream:

        def gen():
            yield from _sse_chunks(final_text, model, pt or 1)

        return StreamingResponse(gen(), media_type="text/event-stream")

    ct = int(usage.get("completion_tokens") or max(1, len(final_text) // 4))
    out = {
        "id": data.get("id") or f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": data.get("created") or int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }
    return JSONResponse(out)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("BRIDGE_HOST", "127.0.0.1")
    port = int(os.environ.get("BRIDGE_PORT", "8765"))
    uvicorn.run(app, host=host, port=port)
