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

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tool_parse import augment_messages_with_tools, has_function_tools, maybe_extract_tool_calls_from_text

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "https://api.ratu.sh").rstrip("/")
BRIDGE_SECRET = os.environ.get("BRIDGE_SECRET", "").strip()
REQUEST_TIMEOUT = float(os.environ.get("BRIDGE_UPSTREAM_TIMEOUT", "120"))

app = FastAPI(title="openclaw-bridge", version="1.0.0")


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

    if tool_calls:
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
