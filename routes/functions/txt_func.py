# routes/functions/txt_func.py

from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import (
    ERROR_HANDLER, 
    calculate_token,
    create_session
)
import json

# Импортируем необходимые константы и функции из других модулей
from .shared_func import (
    format_openai_response,
    stream_response,
    extract_openai_usage_from_1min_metadata
)

# ------------------------------------------------------------------#
# 1min.ai иногда стримит "🌐 Crawling site ..." как часть контента.
# Это служебный трейс: в UI он уходит в Related links, но в OpenAI-like
# прокси его нужно убрать из текста ответа. 
# ------------------------------------------------------------------#

def strip_crawl_prefix(text: str) -> str:
    """
    Убирает служебный префикс провайдера в начале ответа.

    1min.ai / совместимые провайдеры иногда добавляют в контент "🌐 Crawling site ..."
    и/или "tool ... read ... Content loaded." в стиле UI/агента. Это не должно
    попадать в OpenAI-like `assistant.content`.
    """
    if not isinstance(text, str) or not text:
        return text or ""

    lines = text.splitlines(keepends=True)
    out = []
    skipping = True
    expect_tool_args = False

    def _is_crawl_line(line: str) -> bool:
        t = (line or "").lstrip()
        return "crawling site" in t.lower()

    def _is_tool_noise(line: str) -> bool:
        t = (line or "").strip()
        if t == "":
            return True
        if t == "tool":
            return True
        if t in ("memory_search", "read"):
            return True
        if t.lower().startswith("read /"):
            return True
        if t.lower().startswith("(calling "):
            return True
        if t.lower() in ("content loaded.", "content loaded"):
            return True
        # Часто tool-аргументы приходят отдельной JSON-строкой после имени tool.
        if expect_tool_args and t.startswith("{") and t.endswith("}"):
            return True
        return False

    for line in lines:
        if skipping:
            if _is_crawl_line(line):
                continue
            t = (line or "").strip()
            if t in ("memory_search", "read"):
                expect_tool_args = True
                continue
            if expect_tool_args and t.startswith("{") and t.endswith("}"):
                expect_tool_args = False
                continue
            if _is_tool_noise(line):
                continue

            skipping = False
            expect_tool_args = False

        out.append(line)

    return "".join(out).lstrip("\n\r")

#=================================================================#
# ----------- Функции для работы с текстовыми моделями -----------#
#=================================================================#

def format_conversation_history(messages, new_input):
    """
    Formats the conversation history into a structured string.

    Args:
        messages (list): List of message dictionaries from the request
        new_input (str): The new user input message

    Returns:
        str: Formatted conversation history
    """
    formatted_history = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        # Handle potential list content
        if isinstance(content, list):
            processed_content = []
            for item in content:
                if "text" in item:
                    processed_content.append(item["text"])
            content = "\n".join(processed_content)

        if role == "system":
            formatted_history.append(f"System: {content}")
        elif role == "user":
            formatted_history.append(f"User: {content}")
        elif role == "assistant":
            formatted_history.append(f"Assistant: {content}")
        elif role == "tool":
            tool_name = message.get("name") or message.get("tool_name") or "tool"
            formatted_history.append(f"Tool[{tool_name}]: {content}")
        elif role == "function":
            func_name = message.get("name") or "function"
            formatted_history.append(f"Function[{func_name}]: {content}")

    # Add new input if it is
    if new_input:
        formatted_history.append(f"User: {new_input}")

    # We return only the history of dialogue without additional instructions
    return "\n".join(formatted_history)


def _build_tool_calling_instructions(tools: list) -> str:
    """
    Инструкция для провайдера, который НЕ поддерживает нативные tool_calls,
    но способен следовать формату (эмуляция tool calling через JSON в тексте).
    """
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


def _maybe_extract_tool_calls_from_text(text: str):
    """
    Если модель вернула JSON c tool_calls в тексте — выделяем tool_calls для OpenAI-like ответа.
    Возвращает (clean_text, tool_calls_or_None).
    """
    if not isinstance(text, str):
        return "", None
    raw = text.strip()
    if not raw:
        return "", None

    # allow ```json fenced blocks
    if raw.startswith("```"):
        raw2 = raw.strip("`").strip()
        if raw2.lower().startswith("json"):
            raw = raw2[4:].strip()
        else:
            raw = raw2

    if not (raw.startswith("{") and raw.endswith("}")):
        return text, None

    try:
        obj = json.loads(raw)
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
        # OpenAI expects arguments as a JSON string
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

def get_model_capabilities(model, api_key=None, request_id=None):
    """
    Defines supported opportunities for a specific model

    Args:
        Model: The name of the model

    Returns:
        DICT: Dictionary with flags of supporting different features
    """
    capabilities = {
        "vision": False,
        "code_interpreter": False,
        "retrieval": False,
        "function_calling": False,
    }

    # Try dynamic registry (best-effort). If no api_key is provided, fall back to static lists.
    try:
        if api_key:
            from utils.model_registry import get_model_registry_data

            reg = get_model_registry_data(
                api_key=api_key,
                request_id=request_id or "models",
                fallback_model_ids=ALL_ONE_MIN_AVAILABLE_MODELS,
            )
            capabilities["vision"] = model in set(reg.vision_model_ids)
            capabilities["code_interpreter"] = model in set(reg.code_interpreter_model_ids)
            # Upstream doesn't expose retrieval flag per model reliably; keep static allow-list
            capabilities["retrieval"] = model in RETRIEVAL_SUPPORTED_MODELS
            capabilities["function_calling"] = model in FUNCTION_CALLING_SUPPORTED_MODELS
            return capabilities
    except Exception:
        # Never fail hard on capabilities; we'll use static allow-lists below.
        pass

    # Static allow-lists fallback
    capabilities["vision"] = model in VISION_SUPPORTED_MODELS_UNIFY_CHAT_WITH_AI
    capabilities["code_interpreter"] = model in CODE_INTERPRETER_SUPPORTED_MODELS
    capabilities["retrieval"] = model in RETRIEVAL_SUPPORTED_MODELS
    capabilities["function_calling"] = model in FUNCTION_CALLING_SUPPORTED_MODELS

    return capabilities

def prepare_payload(
        request_data, model, all_messages, image_paths=None, file_ids=None, request_id=None
):
    """
    Prepares Payload for request, taking into account the capabilities of the model

    Args:
        Request_Data: Request data
        Model: Model
        All_Messages: Posts of Posts
        image_paths: ways to images
        Request_id: ID query

    Returns:
        DICT: Prepared Payload
    """
    capabilities = get_model_capabilities(model, api_key=request_data.get("_api_key"), request_id=request_id)

    # Check the availability of Openai tools
    tools = request_data.get("tools", [])
    web_search = False
    code_interpreter = False

    if tools:
        for tool in tools:
            tool_type = tool.get("type", "")
            # Trying to include functions, but if they are not supported, we just log in
            if tool_type == "retrieval":
                if capabilities["retrieval"]:
                    web_search = True
                    logger.debug(
                        f"[{request_id}] Enabled web search due to retrieval tool"
                    )
                else:
                    logger.debug(
                        f"[{request_id}] Model {model} does not support web search, ignoring retrieval tool"
                    )
            elif tool_type == "code_interpreter":
                if capabilities["code_interpreter"]:
                    code_interpreter = True
                    logger.debug(f"[{request_id}] Enabled code interpreter")
                else:
                    logger.debug(
                        f"[{request_id}] Model {model} does not support code interpreter, ignoring tool"
                    )
            elif tool_type == "function":
                # Minimal compatibility: accept OpenAI function tools without failing.
                # 1min.ai UNIFY_CHAT_WITH_AI tool-calling is not wired here; we ignore.
                continue
            else:
                logger.debug(f"[{request_id}] Ignoring unsupported tool: {tool_type}")

    # We check the direct parameters 1min.ai
    if not web_search and request_data.get("web_search", False):
        if capabilities["retrieval"]:
            web_search = True
        else:
            logger.debug(
                f"[{request_id}] Model {model} does not support web search, ignoring web_search parameter"
            )

    num_of_site = request_data.get("num_of_site", 3)
    max_word = request_data.get("max_word", 500)

    # Если клиент явно попросил AI Feature API (ONE_MIN_API_URL) через поле type,
    # готовим payload в "features"-формате. Это нужно для CODE_GENERATOR и других feature-эндпоинтов.
    requested_type = (request_data.get("type") or "").strip()
    if requested_type and requested_type != "UNIFY_CHAT_WITH_AI":
        conversation_id = (
            request_data.get("conversation_id")
            or request_data.get("conversationId")
            or request_data.get("conversation")
            or requested_type
        )

        prompt_object = {"prompt": all_messages}
        # features API использует webSearch/numOfSite/maxWord прямо в promptObject
        if web_search:
            prompt_object["webSearch"] = True
            prompt_object["numOfSite"] = num_of_site
            prompt_object["maxWord"] = max_word

        # Некоторые фичи ожидают imageUrl/imageList — не трогаем здесь, т.к. эти маршруты
        # обрабатываются отдельно (images/audio routes). Для CODE_GENERATOR достаточно prompt.
        return {
            "type": requested_type,
            "model": model,
            "conversationId": conversation_id,
            "promptObject": prompt_object,
        }

    # Новый Chat with AI API:
    # - endpoint: /api/chat-with-ai
    # - type: UNIFY_CHAT_WITH_AI (по умолчанию, но указываем явно)
    # - webSearch/settings/history/attachments вложены в promptObject
    conversation_id = (
        request_data.get("conversation_id")
        or request_data.get("conversationId")
        or request_data.get("conversation")
    )

    # If there are images - add instructions for vision-enabled models
    prompt_text = all_messages
    if image_paths and capabilities["vision"]:
        if not prompt_text.strip().startswith(IMAGE_DESCRIPTION_INSTRUCTION):
            prompt_text = f"{IMAGE_DESCRIPTION_INSTRUCTION}\n\n{prompt_text}"

    # If there are files - add instructions for document understanding
    if file_ids:
        if not prompt_text.strip().startswith(DOCUMENT_ANALYSIS_INSTRUCTION):
            prompt_text = f"{DOCUMENT_ANALYSIS_INSTRUCTION}\n\n{prompt_text}"

    # Tools: function-calling emulation for OpenAI-like clients (OpenClaw, etc.)
    tools = request_data.get("tools", []) or []
    if tools and bool(request_data.get("_openclaw")) and capabilities.get("function_calling"):
        instr = _build_tool_calling_instructions(tools)
        if instr:
            prompt_text = f"{instr}\n\n{prompt_text}"

    settings_obj = {}
    # Align with the shape used by OpenClaw 1min.ai plugin: always include webSearchSettings (webSearch defaults to false)
    settings_obj["webSearchSettings"] = {
        "webSearch": bool(web_search),
        "numOfSite": num_of_site,
        "maxWord": max_word,
    }

    # code_interpreter: OpenAI-like clients may request this via tools.
    # 1min.ai doesn't document a stable toggle for UNIFY_CHAT_WITH_AI, but some backends
    # accept it. We only include it when model is allow-listed; if ignored, it's a no-op.
    if code_interpreter and capabilities["code_interpreter"]:
        settings_obj["codeInterpreterSettings"] = {"codeInterpreter": True}

    settings_obj["historySettings"] = {
        "isMixed": bool(request_data.get("is_mixed", False) or request_data.get("isMixed", False)),
        "historyMessageLimit": int(request_data.get("history_message_limit", 10) or request_data.get("historyMessageLimit", 10) or 10),
    }
    # Align with plugin: always include withMemories (defaults to false)
    settings_obj["withMemories"] = bool(request_data.get("with_memories", False) or request_data.get("withMemories", False))

    attachments = {}
    if image_paths:
        attachments["images"] = list(image_paths)
    if file_ids:
        attachments["files"] = list(file_ids)

    prompt_object = {"prompt": prompt_text}
    if conversation_id:
        prompt_object["conversationId"] = conversation_id
    if settings_obj:
        prompt_object["settings"] = settings_obj
    if attachments:
        prompt_object["attachments"] = attachments

    payload = {
        "type": request_data.get("type") or "UNIFY_CHAT_WITH_AI",
        "model": model,
        "promptObject": prompt_object,
    }

    return payload

def transform_response(one_min_response, request_data, prompt_token):
    try:
        # Output of the response structure for debugging
        logger.debug(f"Response structure: {json.dumps(one_min_response)[:200]}...")

        def _coerce_to_text(obj):
            """
            Best-effort extraction of assistant text from 1min.ai response shapes.
            1min.ai responses are not fully stable across endpoints/models.
            """
            if obj is None:
                return ""
            if isinstance(obj, str):
                return obj
            if isinstance(obj, (int, float, bool)):
                return str(obj)
            if isinstance(obj, list):
                for it in obj:
                    t = _coerce_to_text(it)
                    if t:
                        return t
                return ""
            if isinstance(obj, dict):
                # Common direct keys
                for k in (
                    "text",
                    "content",
                    "result",
                    "message",
                    "output",
                    "answer",
                    "response",
                ):
                    if k in obj:
                        t = _coerce_to_text(obj.get(k))
                        if t:
                            return t

                # OpenAI-like shapes (best-effort)
                choices = obj.get("choices")
                if isinstance(choices, list) and choices:
                    c0 = choices[0] if isinstance(choices[0], dict) else {}
                    msg = c0.get("message") if isinstance(c0, dict) else None
                    if isinstance(msg, dict):
                        t = _coerce_to_text(msg.get("content"))
                        if t:
                            return t
                    t = _coerce_to_text(c0.get("text") if isinstance(c0, dict) else "")
                    if t:
                        return t

                # Some providers embed payloads under data/responseObject/resultObject
                for k in ("responseObject", "resultObject", "data"):
                    if k in obj:
                        t = _coerce_to_text(obj.get(k))
                        if t:
                            return t

                return ""

            return ""

        detail = (one_min_response.get("aiRecord") or {}).get("aiRecordDetail") or {}
        # Primary: aiRecord.aiRecordDetail.resultObject (documented in older 1min.ai payloads)
        result_text = _coerce_to_text(detail.get("resultObject"))
        # Fallbacks seen in newer payloads / some providers
        if not result_text:
            result_text = _coerce_to_text(detail.get("responseObject"))
        if not result_text:
            result_text = _coerce_to_text(one_min_response.get("resultObject"))
        if not result_text:
            result_text = _coerce_to_text(one_min_response.get("result"))
        if not result_text:
            result_text = _coerce_to_text(one_min_response.get("responseObject"))

        if not result_text:
            # Alternative ways to extract an answer
            if "resultObject" in one_min_response:
                result_text = (
                    one_min_response["resultObject"][0]
                    if isinstance(one_min_response["resultObject"], list)
                    else one_min_response["resultObject"]
                )
            elif "result" in one_min_response:
                result_text = one_min_response["result"]
            else:
                # If you have not found an answer along the well -known paths, we return the error
                logger.error(f"Cannot extract response text from API result")
                result_text = "Error: Could not extract response from API"

        # In OpenClaw tool-calling mode, 1min.ai may return tool traces like:
        #   tool\nwrite\n{...}\nContent loaded.
        # Our `strip_crawl_prefix()` would remove these lines and make the content empty.
        # So: first try to extract tool_calls from the raw text; only then strip crawl/tool noise.
        raw_text = result_text

        clean_text, tool_calls = _maybe_extract_tool_calls_from_text(raw_text) if request_data.get("_openclaw") else (raw_text, None)
        if not tool_calls:
            # Убираем служебный префикс 1min.ai (crawl trace / agent noise), если он попал в текст.
            result_text = strip_crawl_prefix(raw_text)
            # Tool-calling emulation: convert JSON tool_calls in text -> OpenAI tool_calls field.
            clean_text, tool_calls = _maybe_extract_tool_calls_from_text(result_text)

        completion_token = calculate_token(clean_text)
        logger.debug(
            f"Finished processing Non-Streaming response. Completion tokens: {str(completion_token)}"
        )
        logger.debug(f"Total tokens: {str(completion_token + prompt_token)}")

        usage = extract_openai_usage_from_1min_metadata(one_min_response) or {
            "prompt_tokens": prompt_token,
            "completion_tokens": completion_token,
            "total_tokens": prompt_token + completion_token,
        }

        finish_reason = "tool_calls" if tool_calls else "stop"
        message_obj = {"role": "assistant", "content": clean_text}
        if tool_calls:
            message_obj["tool_calls"] = tool_calls

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "mistral-nemo").strip(),
            "choices": [
                {
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
    except Exception as e:
        logger.error(f"Error in transform_response: {str(e)}")
        # Return the error in the format compatible with Openai
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "mistral-nemo").strip(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Error processing response: {str(e)}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token,
                "completion_tokens": 0,
                "total_tokens": prompt_token,
            },
        }

def emulate_stream_response(full_content, request_data, model, prompt_tokens):
    """
    Emulates a streaming response for cases when the API does not support the flow gear

    Args:
        Full_Content: Full text of the answer
        Request_Data: Request data
        Model: Model
        Prompt_tokens: the number of tokens in the request

    Yields:
        STR: Lines for streaming
    """
    # We break the answer to fragments by ~ 5 words
    words = full_content.split()
    chunks = [" ".join(words[i: i + 5]) for i in range(0, len(words), 5)]

    for chunk in chunks:
        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
            ],
        }

        yield f"data: {json.dumps(return_chunk)}\n\n"
        time.sleep(0.05)  # Small delay in emulating stream

    # We calculate the tokens
    tokens = calculate_token(full_content)

    # Final chambers
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens + prompt_tokens,
        },
    }

    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
# Определяем функцию streaming_request, которая используется для обработки потоковых запросов
def streaming_request(api_url, payload, headers, request_id, model, model_settings=None, api_params=None):
    """
    Выполняет потоковый запрос к API и возвращает ответ в формате потока
    
    Args:
        api_url: URL для запроса
        payload: Данные запроса
        headers: Заголовки запроса
        request_id: ID запроса для логирования
        model: Модель для запроса
        model_settings: Настройки модели (опционально)
        api_params: Дополнительные параметры запроса (опционально)
        
    Returns:
        Response: Объект потокового ответа
    """
    try:
        # Используем сессию для контроля соединения
        session = create_session()
        
        # Добавляем параметры запроса, если они есть
        if api_params:
            response_stream = session.post(
                api_url, json=payload, headers=headers, params=api_params, stream=True
            )
        else:
            response_stream = session.post(
                api_url, json=payload, headers=headers, stream=True
            )
        
        logger.debug(f"[{request_id}] Streaming response status code: {response_stream.status_code}")
        
        if response_stream.status_code != 200:
            if response_stream.status_code == 401:
                session.close()
                return ERROR_HANDLER(1020, key=headers.get("API-KEY", ""))
                
            logger.error(f"[{request_id}] Error status code: {response_stream.status_code}")
            try:
                error_content = response_stream.json()
                logger.error(f"[{request_id}] Error response: {error_content}")
            except:
                logger.error(f"[{request_id}] Could not parse error response as JSON")
                
            session.close()
            return ERROR_HANDLER(response_stream.status_code)
            
        # Вычисляем количество токенов
        prompt_token = calculate_token(payload["message"] if "message" in payload else 
                                      payload.get("promptObject", {}).get("prompt", ""))
        
        # Передаем сессию генератору
        return Response(
            stream_response(response_stream, {"model": model}, model, prompt_token, session),
            content_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Exception during streaming request: {str(e)}")
        return jsonify({"error": str(e)}), 500
