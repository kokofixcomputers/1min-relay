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
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return None, ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
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
    
    # Обрабатываем контент
    for chunk in response.iter_content(chunk_size=1024):
        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk.decode('utf-8')},
                "finish_reason": None
            }]
        }
        all_chunks += chunk.decode('utf-8')
        yield f"data: {json.dumps(return_chunk)}\n\n"
        
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
    
    if session:
        session.close() 
