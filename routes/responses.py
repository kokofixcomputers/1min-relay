# version 1.0.0
# OpenAI Responses API (best-effort compatibility)

from utils.imports import *
from utils.logger import logger
from utils.constants import (
    ONE_MIN_API_URL,
    ONE_MIN_CHAT_WITH_AI_URL,
    PERMIT_MODELS_FROM_SUBSET_ONLY,
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS,
)
from utils.common import (
    ERROR_HANDLER,
    handle_options_request,
    set_response_headers,
    calculate_token,
    api_request_with_websearch_degradation,
)
from routes import app, limiter
from .functions import validate_auth, format_conversation_history, prepare_payload, transform_response


def _build_response_format_instructions(response_format: dict | None) -> str | None:
    if not isinstance(response_format, dict):
        return None
    rtype = response_format.get("type")
    if rtype == "json_object":
        return (
            "IMPORTANT: Output MUST be a single valid JSON object. "
            "Do not wrap it in markdown, do not include any extra text."
        )
    if rtype == "json_schema":
        schema = response_format.get("json_schema") or {}
        return (
            "IMPORTANT: Output MUST be a single valid JSON object that matches this JSON Schema. "
            "Do not wrap it in markdown, do not include any extra text.\n\n"
            f"JSON_SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}"
        )
    return None


def _build_reasoning_effort_instructions(reasoning_effort: str | None) -> str | None:
    if reasoning_effort not in ("low", "medium", "high"):
        return None
    return f"Reasoning effort: {reasoning_effort}. Keep the response correct and concise."


def _responses_api_shape_from_chat_completion(chat_obj: dict) -> dict:
    choice = (chat_obj.get("choices") or [{}])[0] or {}
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []

    content_items = [{"type": "output_text", "text": text}]
    if isinstance(tool_calls, list) and tool_calls:
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            content_items.append(
                {
                    "type": "tool_call",
                    "tool_call_id": tc.get("id"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                }
            )

    created = chat_obj.get("created") or int(time.time())
    model = chat_obj.get("model") or ""
    return {
        "id": f"resp-{uuid.uuid4()}",
        "object": "response",
        "created": created,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": content_items,
            }
        ],
        "output_text": text,
        "usage": chat_obj.get("usage") or {},
    }


@app.route("/v1/responses", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def responses():
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info("[%s] Received request: /v1/responses", request_id)

    api_key, auth_err = validate_auth(request, request_id)
    if auth_err:
        return auth_err

    if not request.json:
        return jsonify({"error": "Invalid request format"}), 400

    # Responses API is non-streaming in our implementation (as in worker reference)
    request_data = request.json.copy()
    request_data["stream"] = False
    request_data["_api_key"] = api_key

    model = (request_data.get("model") or "").strip() or "gpt-4o-mini"
    messages = request_data.get("messages")
    input_text = request_data.get("input")

    if not messages and not input_text:
        return jsonify({"error": {"message": "Provide either 'input' or 'messages'."}}), 400

    # Normalize to OpenAI chat-style messages for reuse of existing pipeline
    if not messages:
        messages = [{"role": "user", "content": str(input_text)}]

    # Inject best-effort instructions for structured outputs / reasoning effort
    instruction_chunks = []
    instruction_chunks.append(_build_response_format_instructions(request_data.get("response_format")))
    instruction_chunks.append(_build_reasoning_effort_instructions(request_data.get("reasoning_effort")))
    instruction = "\n\n".join([c for c in instruction_chunks if c])
    if instruction:
        messages = [{"role": "system", "content": instruction}] + list(messages)

    # Build payload like /v1/chat/completions
    all_messages = format_conversation_history(messages, "")
    prompt_token = calculate_token(all_messages, model=model)

    # Respect model restriction if enabled (same semantics as /v1/chat/completions)
    if PERMIT_MODELS_FROM_SUBSET_ONLY and model not in SUBSET_OF_ONE_MIN_PERMITTED_MODELS:
        return ERROR_HANDLER(1002, model=model)

    payload = prepare_payload(
        {"messages": messages, "model": model, "_api_key": api_key},
        model,
        all_messages,
        request_id=request_id,
    )

    headers = {"API-KEY": api_key, "Content-Type": "application/json"}
    requested_type = (payload.get("type") or "").strip()
    api_url = (
        ONE_MIN_CHAT_WITH_AI_URL
        if (not requested_type or requested_type == "UNIFY_CHAT_WITH_AI")
        else ONE_MIN_API_URL
    )

    response, degraded = api_request_with_websearch_degradation("POST", api_url, json=payload, headers=headers)
    if response.status_code != 200:
        if response.status_code == 401:
            return ERROR_HANDLER(1020, key=api_key)
        return ERROR_HANDLER(response.status_code)

    one_min_response = response.json()
    chat_obj = transform_response(one_min_response, {"model": model, "messages": messages}, prompt_token)
    resp_obj = _responses_api_shape_from_chat_completion(chat_obj)

    flask_resp = make_response(jsonify(resp_obj))
    set_response_headers(flask_resp)
    if degraded:
        flask_resp.headers["X-WebSearch-Degraded"] = "true"
    return flask_resp, 200

