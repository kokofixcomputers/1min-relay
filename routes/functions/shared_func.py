# routes/functions/shared_func.py

from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import (
    ERROR_HANDLER, 
    handle_options_request, 
    set_response_headers, 
    create_session, 
    api_request, 
    safe_temp_file, 
    calculate_token
)
from utils.memcached import safe_memcached_operation

#========================================================================#
# ----------- Общие функции для авторизации и обработки ошибок ----------#
#========================================================================#

def validate_auth(request, request_id=None):
    """
    Проверяет авторизацию запроса
    
    Args:
        request: Объект запроса Flask
        request_id: ID запроса для логирования
    
    Returns:
        tuple: (api_key, error_response)
        api_key будет None если авторизация не прошла
    """
    # Поддерживаем оба формата:
    # - OpenAI-like клиенты обычно шлют: Authorization: Bearer <key>
    # - 1min.ai upstream использует: API-KEY: <key>
    auth_header = request.headers.get("Authorization") or ""
    api_key_header = request.headers.get("API-KEY") or request.headers.get("Api-Key") or ""

    api_key = None
    if auth_header.startswith("Bearer "):
        api_key = auth_header.split(" ", 1)[1].strip() or None
    elif api_key_header:
        api_key = str(api_key_header).strip() or None

    if not api_key:
        logger.error(f"[{request_id}] Invalid Authentication")
        return None, ERROR_HANDLER(1021)

    return api_key, None

def handle_api_error(response, api_key=None, request_id=None):
    """
    Обрабатывает ошибки API
    
    Args:
        response: Ответ от API
        api_key: API ключ пользователя
        request_id: ID запроса для логирования
    
    Returns:
        tuple: (error_json, status_code)
    """
    if response.status_code == 401:
        return ERROR_HANDLER(1020, key=api_key)
    
    error_text = "Unknown error"
    try:
        error_json = response.json()
        if "error" in error_json:
            error_text = error_json["error"]
    except:
        pass
    
    logger.error(f"[{request_id}] API error: {response.status_code} - {error_text}")
    return jsonify({"error": error_text}), response.status_code

#=======================================================#
# ----------- Функции форматирования ответов -----------#
#=======================================================#

def format_openai_response(content, model, request_id=None, prompt_tokens=0):
    """
    Форматирует ответ в формат OpenAI API
    
    Args:
        content: Текст ответа
        model: Название модели
        request_id: ID запроса для логирования
        prompt_tokens: Количество токенов в запросе
    
    Returns:
        dict: Ответ в формате OpenAI
    """
    completion_tokens = calculate_token(content)
    
    return {
        "id": f"chatcmpl-{request_id or str(uuid.uuid4())[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

def format_image_response(image_urls, request_id=None, model=None):
    """
    Форматирует ответ с изображениями в формат OpenAI API
    
    Args:
        image_urls: Список URL изображений
        request_id: ID запроса для логирования
        model: Название модели
        
    Returns:
        dict: Ответ в формате OpenAI
    """
    # Формируем полные URL для отображения
    full_urls = []
    asset_host = "https://asset.1min.ai"
    
    for url in image_urls:
        if not url:
            continue
            
        if not url.startswith("http"):
            if url.startswith("/"):
                full_url = f"{asset_host}{url}"
            else:
                full_url = f"{asset_host}/{url}"
        else:
            full_url = url
            
        full_urls.append(full_url)
        
    # Формируем ответ в формате OpenAI
    openai_data = []
    for i, url in enumerate(full_urls, 1):
        openai_data.append({
            "url": url,
            "revised_prompt": None,
            "variation_commands": {
                "variation": f"/v{i} {url}"
            } if model in IMAGE_VARIATION_MODELS else None
        })
        
    # Формируем текст с кнопками вариаций
    markdown_text = ""
    if len(full_urls) == 1:
        markdown_text = f"![Image]({full_urls[0]}) `[_V1_]`"
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** and send it (paste) in the next **prompt**"
    else:
        image_lines = []
        for i, url in enumerate(full_urls, 1):
            image_lines.append(f"![Image {i}]({url}) `[_V{i}_]`")
            
        markdown_text = "\n".join(image_lines)
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**"
        
    response = {
        "created": int(time.time()),
        "data": openai_data,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": markdown_text,
                "structured_output": {
                    "type": "image",
                    "image_urls": full_urls
                }
            },
            "index": 0,
            "finish_reason": "stop"
        }]
    }
    
    logger.debug(f"[{request_id}] Formatted response with {len(full_urls)} images")
    return response 

def stream_response(response, request_data, model, prompt_tokens, session=None):
    """
    Стримит ответ от API в формате OpenAI
    
    Args:
        response: Ответ от API
        request_data: Данные запроса
        model: Название модели
        prompt_tokens: Количество токенов в запросе
        session: Сессия для запросов
    
    Yields:
        str: Строки для стриминга
    """
    all_chunks = ""
    session_created = False
    
    # Если сессия не передана, создаем новую
    if not session:
        session = create_session()
        session_created = True
    
    # Отправляем первый фрагмент с ролью
    first_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    
    yield f"data: {json.dumps(first_chunk)}\n\n"
    
    def _is_crawl_line(s: str) -> bool:
        if not s:
            return False
        t = s.lstrip()
        # "ðŸŒ" — это тот же 🌐, если где-то пошла неверная декодировка.
        return t.startswith("🌐 Crawling site") or t.startswith("ðŸŒ Crawling site")

    def _is_tool_prefix_noise(s: str, expect_tool_args: bool) -> bool:
        """
        1min.ai иногда возвращает в content "tool ...", "read ...", "Content loaded."
        и даже JSON-аргументы тулов как текст. Мы гасим это только в самом начале ответа.
        """
        if s is None:
            return False
        t = str(s).strip()
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
        if expect_tool_args and t.startswith("{") and t.endswith("}"):
            return True
        return False

    def _emit_delta(text: str):
        nonlocal all_chunks
        if not text:
            return
        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None
            }]
        }
        all_chunks += text
        yield f"data: {json.dumps(return_chunk)}\n\n"

    try:
        # 1min.ai иногда возвращает SSE вида:
        # event: content
        # data: {"content":"..."}
        # Поэтому сначала пытаемся распарсить как SSE и извлечь поле content.
        buffer = ""
        seen_sse = False
        skipping_prefix = True
        expect_tool_args = False

        for raw in response.iter_content(chunk_size=1024):
            if not raw:
                continue
            part = raw.decode("utf-8", errors="replace")
            buffer += part

            # Быстро определяем, что это SSE.
            if not seen_sse and ("event:" in buffer or "data:" in buffer):
                seen_sse = True

            if not seen_sse:
                # Фоллбек: API шлёт просто текст.
                if skipping_prefix:
                    if _is_crawl_line(part):
                        continue
                    if _is_tool_prefix_noise(part, expect_tool_args):
                        if part.strip() in ("memory_search", "read"):
                            expect_tool_args = True
                        elif expect_tool_args and part.strip().startswith("{") and part.strip().endswith("}"):
                            expect_tool_args = False
                        continue
                    skipping_prefix = False
                    expect_tool_args = False
                yield from _emit_delta(part)
                continue

            # SSE: обрабатываем завершённые блоки, разделённые пустой строкой.
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                if not block.strip():
                    continue

                event_name = None
                data_lines = []
                for line in block.splitlines():
                    if line.startswith("event:"):
                        event_name = line[6:].strip() or None
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if not data_lines:
                    continue

                data = "\n".join(data_lines).strip()
                if data == "[DONE]":
                    continue

                # Новый Chat with AI API использует события: content/result/done/error.
                # - content: {"content": "..."}
                # - result:  {"aiRecord": {...}}
                # - done:    {"message":"Stream completed"}
                # - error:   {"success":false,"error":{...}} (или иной формат)
                if event_name == "done":
                    buffer = ""
                    break

                if event_name == "error":
                    err_text = None
                    if data.startswith("{") and data.endswith("}"):
                        try:
                            obj = json.loads(data)
                            if isinstance(obj, dict):
                                if isinstance(obj.get("error"), dict):
                                    err_text = obj["error"].get("message") or obj["error"].get("code")
                                err_text = err_text or obj.get("message") or obj.get("error")
                        except Exception:
                            err_text = None
                    err_text = err_text or data or "Unknown error"
                    yield from _emit_delta(f"Error: {err_text}")
                    buffer = ""
                    break

                # Пытаемся распарсить JSON с content.
                content = None
                if data.startswith("{") and data.endswith("}"):
                    try:
                        obj = json.loads(data)
                        if isinstance(obj, dict):
                            content = obj.get("content")
                    except Exception:
                        content = None

                if content is None:
                    # Если не JSON — считаем, что это текст.
                    content = data

                if not isinstance(content, str):
                    continue

                # Гасим служебный префикс только в начале ответа.
                if skipping_prefix:
                    if _is_crawl_line(content):
                        continue
                    if _is_tool_prefix_noise(content, expect_tool_args):
                        t = content.strip()
                        if t in ("memory_search", "read"):
                            expect_tool_args = True
                        elif expect_tool_args and t.startswith("{") and t.endswith("}"):
                            expect_tool_args = False
                        continue
                    skipping_prefix = False
                    expect_tool_args = False

                yield from _emit_delta(content)

        # Остаток буфера (если вдруг API оборвалось без \n\n)
        if seen_sse and buffer.strip():
            # Пытаемся вытащить последнюю data:... строку
            for line in buffer.splitlines():
                if line.startswith("data:"):
                    tail = line[5:].lstrip()
                    if tail and tail != "[DONE]":
                        try:
                            obj = json.loads(tail)
                            tail_content = obj.get("content") if isinstance(obj, dict) else None
                        except Exception:
                            tail_content = tail
                        if isinstance(tail_content, str):
                            if skipping_prefix:
                                if _is_crawl_line(tail_content):
                                    continue
                                if _is_tool_prefix_noise(tail_content, expect_tool_args):
                                    t = tail_content.strip()
                                    if t in ("memory_search", "read"):
                                        expect_tool_args = True
                                    elif expect_tool_args and t.startswith("{") and t.endswith("}"):
                                        expect_tool_args = False
                                    continue
                                skipping_prefix = False
                                expect_tool_args = False
                            yield from _emit_delta(tail_content)
            buffer = ""
        
        # Считаем токены
        tokens = calculate_token(all_chunks)
        
        # Финальный чанк
        final_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": ""},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": tokens + prompt_tokens
            }
        }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    except requests.exceptions.ChunkedEncodingError:
        # Обрабатываем ошибку прерванного соединения
        logger.warning(f"Соединение с API прервано преждевременно. Получена только часть ответа.")
        error_message = "Соединение прервано. Получена только часть ответа."
        
        # Считаем токены для полученной части ответа
        tokens = calculate_token(all_chunks)
        
        # Отправляем уведомление об ошибке клиенту
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n{error_message}"},
                "finish_reason": "error"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": tokens + prompt_tokens
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    except Exception as e:
        # Обрабатываем другие возможные исключения
        logger.error(f"Ошибка при потоковой передаче: {str(e)}")
        error_message = f"Ошибка при получении ответа: {str(e)}"
        
        # Считаем токены для полученной части ответа
        tokens = calculate_token(all_chunks)
        
        # Отправляем уведомление об ошибке клиенту
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n{error_message}"},
                "finish_reason": "error"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": tokens + prompt_tokens
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    finally:
        # Закрываем сессию, если она была создана внутри функции
        if session_created and session:
            session.close()

#=======================================================#
# ----------- Функции извлечения данных из API ---------#
#=======================================================#

def get_full_url(url, asset_host="https://asset.1min.ai"):
    """
    Формирует полный URL для ресурса
    
    Args:
        url: Относительный или абсолютный URL
        asset_host: Базовый URL хоста
        
    Returns:
        str: Полный URL
    """
    if not url.startswith("http"):
        return f"{asset_host}{url}" if url.startswith("/") else f"{asset_host}/{url}"
    return url

def extract_data_from_api_response(response_data, request_id=None):
    """
    Общая функция для извлечения данных из ответа API 1min.ai
    
    Args:
        response_data: Данные ответа от API
        request_id: ID запроса для логирования
        
    Returns:
        object: Извлеченный объект данных или None
    """
    try:
        # Проверяем структуру aiRecord (основная структура ответа)
        if "aiRecord" in response_data and "aiRecordDetail" in response_data["aiRecord"]:
            result_object = response_data["aiRecord"]["aiRecordDetail"].get("resultObject", None)
            return result_object
                
        # Проверяем прямую структуру resultObject
        elif "resultObject" in response_data:
            return response_data["resultObject"]
        
        # Ничего не найдено
        logger.error(f"[{request_id}] Could not extract data from API response")
        return None
            
    except Exception as e:
        logger.error(f"[{request_id}] Error extracting data from response: {str(e)}")
        return None

def extract_text_from_response(response_data, request_id=None):
    """
    Извлекает текст из ответа API
    
    Args:
        response_data: Данные ответа от API
        request_id: ID запроса для логирования
        
    Returns:
        str: Извлеченный текст или пустая строка в случае ошибки
    """
    result_text = ""
    
    try:
        # Получаем данные через общую функцию
        result_object = extract_data_from_api_response(response_data, request_id)
        
        if result_object:
            # Обработка в зависимости от типа данных
            if isinstance(result_object, list) and result_object:
                result_text = result_object[0]
            elif isinstance(result_object, str):
                result_text = result_object
            else:
                result_text = str(result_object)
        
        # Проверяем если result_text это json
        if result_text and isinstance(result_text, str) and result_text.strip().startswith("{"):
            try:
                parsed_json = json.loads(result_text)
                if "text" in parsed_json:
                    result_text = parsed_json["text"]
                    logger.debug(f"[{request_id}] Extracted inner text from JSON")
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        
        if not result_text:
            logger.error(f"[{request_id}] Could not extract text from API response")
            
    except Exception as e:
        logger.error(f"[{request_id}] Error extracting text from response: {str(e)}")
        
    return result_text

def extract_image_urls(response_data, request_id=None):
    """
    Извлекает URL изображений из ответа API
    
    Args:
        response_data: Ответ от API
        request_id: ID запроса для логирования
        
    Returns:
        list: Список URL изображений
    """
    image_urls = []
    
    try:
        # Получаем данные через общую функцию
        result_object = extract_data_from_api_response(response_data, request_id)
        
        if result_object:
            # Обработка в зависимости от типа данных
            if isinstance(result_object, list):
                image_urls.extend(result_object)
            elif isinstance(result_object, str):
                image_urls.append(result_object)
                
        # Специфичная проверка для OpenAI-совместимых ответов
        elif "data" in response_data and isinstance(response_data["data"], list):
            for item in response_data["data"]:
                if "url" in item:
                    image_urls.append(item["url"])
                    
        logger.debug(f"[{request_id}] Extracted {len(image_urls)} image URLs")
        
        if not image_urls:
            logger.error(f"[{request_id}] Could not extract image URLs from API response: {json.dumps(response_data)[:500]}")
            
    except Exception as e:
        logger.error(f"[{request_id}] Error extracting image URLs: {str(e)}")
        
    return image_urls

def extract_audio_url(response_data, request_id=None):
    """
    Извлекает URL аудио из ответа API
    
    Args:
        response_data: Данные ответа от API
        request_id: ID запроса для логирования
        
    Returns:
        str: URL аудио или пустая строка в случае ошибки
    """
    audio_url = ""
    
    try:
        # Получаем данные через общую функцию
        result_object = extract_data_from_api_response(response_data, request_id)
        
        if result_object:
            # Обработка в зависимости от типа данных
            if isinstance(result_object, list) and result_object:
                audio_url = result_object[0]
            elif isinstance(result_object, str):
                audio_url = result_object
        
        if not audio_url:
            logger.error(f"[{request_id}] Could not extract audio URL from API response")
            
    except Exception as e:
        logger.error(f"[{request_id}] Error extracting audio URL from response: {str(e)}")
        
    return audio_url 
