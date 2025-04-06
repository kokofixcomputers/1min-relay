# routes/utils.py
# Общие утилиты для маршрутов

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

def get_user_files(api_key, request_id=None):
    """
    Получает список файлов пользователя из Memcached
    
    Args:
        api_key: API ключ пользователя
        request_id: ID запроса для логирования
    
    Returns:
        list: Список файлов пользователя
    """
    user_files = []
    try:
        user_key = f"user:{api_key}"
        user_files_json = safe_memcached_operation('get', user_key)
        
        if user_files_json:
            if isinstance(user_files_json, str):
                user_files = json.loads(user_files_json)
            elif isinstance(user_files_json, bytes):
                user_files = json.loads(user_files_json.decode('utf-8'))
            else:
                user_files = user_files_json
                
            logger.debug(f"[{request_id}] Found {len(user_files)} files for user")
    except Exception as e:
        logger.error(f"[{request_id}] Error getting user files: {str(e)}")
        
    return user_files

def save_user_files(api_key, files, request_id=None):
    """
    Сохраняет список файлов пользователя в Memcached
    
    Args:
        api_key: API ключ пользователя
        files: Список файлов для сохранения
        request_id: ID запроса для логирования
    """
    try:
        user_key = f"user:{api_key}"
        safe_memcached_operation('set', user_key, json.dumps(files))
        logger.debug(f"[{request_id}] Saved {len(files)} files for user")
        
        # Добавляем пользователя в список известных пользователей
        known_users = safe_memcached_operation('get', 'known_users_list') or []
        if isinstance(known_users, str):
            known_users = json.loads(known_users)
        elif isinstance(known_users, bytes):
            known_users = json.loads(known_users.decode('utf-8'))
            
        if api_key not in known_users:
            known_users.append(api_key)
            safe_memcached_operation('set', 'known_users_list', json.dumps(known_users))
            logger.debug(f"[{request_id}] Added user to known_users_list")
    except Exception as e:
        logger.error(f"[{request_id}] Error saving user files: {str(e)}")

def create_temp_file(file_data, suffix=".tmp", request_id=None):
    """
    Создает временный файл с данными
    
    Args:
        file_data: Данные для записи в файл
        suffix: Расширение файла
        request_id: ID запроса для логирования
    
    Returns:
        str: Путь к временному файлу
    """
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(file_data)
        temp_file.close()
        logger.debug(f"[{request_id}] Created temporary file: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"[{request_id}] Error creating temporary file: {str(e)}")
        return None 

def upload_asset(file_data, filename, mime_type, api_key, request_id=None, file_type=None):
    """
    Загружает файл в 1min.ai
    
    Args:
        file_data: Бинарные данные файла
        filename: Имя файла
        mime_type: MIME тип файла
        api_key: API ключ пользователя
        request_id: ID запроса для логирования
        file_type: Тип файла (DOC/DOCX) для специальной обработки
        
    Returns:
        tuple: (asset_id, asset_path, error_response)
    """
    try:
        session = create_session()
        headers = {"API-KEY": api_key}
        
        if file_type:
            headers["X-File-Type"] = file_type
            
        files = {"asset": (filename, file_data, mime_type)}
        
        response = session.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
        logger.debug(f"[{request_id}] Asset upload response status code: {response.status_code}")
        
        if response.status_code != 200:
            error = response.json().get("error", "Failed to upload asset")
            return None, None, (jsonify({"error": error}), response.status_code)
            
        response_data = response.json()
        
        # Извлекаем ID и путь файла
        asset_id = None
        asset_path = None
        
        if "id" in response_data:
            asset_id = response_data["id"]
        elif "fileContent" in response_data:
            if "id" in response_data["fileContent"]:
                asset_id = response_data["fileContent"]["id"]
            elif "uuid" in response_data["fileContent"]:
                asset_id = response_data["fileContent"]["uuid"]
                
            if "path" in response_data["fileContent"]:
                asset_path = response_data["fileContent"]["path"]
                
        if not asset_id and not asset_path:
            return None, None, (jsonify({"error": "Could not extract asset information"}), 500)
            
        logger.debug(f"[{request_id}] Successfully uploaded asset: id={asset_id}, path={asset_path}")
        return asset_id, asset_path, None
        
    except Exception as e:
        logger.error(f"[{request_id}] Error uploading asset: {str(e)}")
        return None, None, (jsonify({"error": str(e)}), 500)
    finally:
        session.close()

def get_mime_type(filename):
    """
    Определяет MIME тип файла по расширению
    
    Args:
        filename: Имя файла
        
    Returns:
        tuple: (mime_type, file_type)
        file_type будет None для всех файлов кроме DOC/DOCX
    """
    extension = os.path.splitext(filename)[1].lower()
    
    mime_types = {
        ".pdf": ("application/pdf", None),
        ".txt": ("text/plain", None),
        ".doc": ("application/msword", "DOC"),
        ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "DOCX"),
        ".csv": ("text/csv", None),
        ".xls": ("application/vnd.ms-excel", None),
        ".xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", None),
        ".json": ("application/json", None),
        ".md": ("text/markdown", None),
        ".html": ("text/html", None),
        ".htm": ("text/html", None),
        ".xml": ("application/xml", None),
        ".pptx": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", None),
        ".ppt": ("application/vnd.ms-powerpoint", None),
        ".rtf": ("application/rtf", None),
        ".png": ("image/png", None),
        ".jpg": ("image/jpeg", None),
        ".jpeg": ("image/jpeg", None),
        ".gif": ("image/gif", None),
        ".webp": ("image/webp", None),
        ".mp3": ("audio/mpeg", None),
        ".wav": ("audio/wav", None),
        ".ogg": ("audio/ogg", None),
    }
    
    return mime_types.get(extension, ("application/octet-stream", None))

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
        # Проверяем структуру aiRecord
        if "aiRecord" in response_data and "aiRecordDetail" in response_data["aiRecord"]:
            result = response_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
            if isinstance(result, list):
                image_urls.extend(result)
            elif isinstance(result, str):
                image_urls.append(result)
                
        # Проверяем прямую структуру resultObject
        elif "resultObject" in response_data:
            result = response_data["resultObject"]
            if isinstance(result, list):
                image_urls.extend(result)
            elif isinstance(result, str):
                image_urls.append(result)
                
        # Проверяем структуру data.url (для Dall-E)
        elif "data" in response_data and isinstance(response_data["data"], list):
            for item in response_data["data"]:
                if "url" in item:
                    image_urls.append(item["url"])
                    
        logger.debug(f"[{request_id}] Extracted {len(image_urls)} image URLs")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error extracting image URLs: {str(e)}")
        
    return image_urls

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

def prepare_image_payload(model, prompt, request_data, image_paths=None, request_id=None):
    """
    Подготавливает payload для запроса генерации изображения
    
    Args:
        model: Название модели
        prompt: Текст запроса
        request_data: Данные запроса
        image_paths: Пути к изображениям
        request_id: ID запроса для логирования
        
    Returns:
        dict: Подготовленный payload
    """
    # Извлекаем параметры из prompt
    aspect_ratio = None
    size = request_data.get("size", "1024x1024")
    mode = None
    negative_prompt = None
    
    # Ищем параметры режима в тексте
    mode_match = re.search(r'(--|\u2014)(fast|relax)\s*', prompt)
    if mode_match:
        mode = mode_match.group(2)
        prompt = re.sub(r'(--|\u2014)(fast|relax)\s*', '', prompt).strip()
        logger.debug(f"[{request_id}] Extracted mode from prompt: {mode}")
        
    # Ищем соотношение сторон
    ar_match = re.search(r'(--|\u2014)ar\s+(\d+):(\d+)', prompt)
    if ar_match:
        width = int(ar_match.group(2))
        height = int(ar_match.group(3))
        aspect_ratio = f"{width}:{height}"
        prompt = re.sub(r'(--|\u2014)ar\s+\d+:\d+\s*', '', prompt).strip()
        logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_ratio}")
        
    # Ищем негативный промпт
    no_match = re.search(r'(--|\u2014)no\s+(.*?)(?=(--|\u2014)|$)', prompt)
    if no_match:
        negative_prompt = no_match.group(2).strip()
        prompt = re.sub(r'(--|\u2014)no\s+.*?(?=(--|\u2014)|$)', '', prompt).strip()
        logger.debug(f"[{request_id}] Extracted negative prompt: {negative_prompt}")
        
    # Формируем базовый payload
    if image_paths:
        payload = {
            "type": "CHAT_WITH_IMAGE",
            "model": model,
            "promptObject": {
                "prompt": prompt,
                "isMixed": False,
                "imageList": image_paths,
            }
        }
    else:
        if model == "dall-e-3":
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": model,
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size,
                    "quality": request_data.get("quality", "standard"),
                    "style": request_data.get("style", "vivid"),
                }
            }
        elif model == "dall-e-2":
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": model,
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size,
                }
            }
        elif model.startswith("midjourney"):
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": model,
                "promptObject": {
                    "prompt": prompt,
                    "mode": mode or request_data.get("mode", "fast"),
                    "n": 4,
                    "aspect_width": int(aspect_ratio.split(":")[0]) if aspect_ratio else 1,
                    "aspect_height": int(aspect_ratio.split(":")[1]) if aspect_ratio else 1,
                    "isNiji6": request_data.get("isNiji6", False),
                    "maintainModeration": request_data.get("maintainModeration", True),
                }
            }
            if negative_prompt:
                payload["promptObject"]["negativePrompt"] = negative_prompt
        else:
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": model,
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                }
            }
            
            if aspect_ratio:
                payload["promptObject"]["aspect_ratio"] = aspect_ratio
            if negative_prompt:
                payload["promptObject"]["negativePrompt"] = negative_prompt
                
    logger.debug(f"[{request_id}] Prepared payload for model {model}")
    return payload

def prepare_chat_payload(model, messages, request_data, request_id=None):
    """
    Подготавливает payload для запроса чата
    
    Args:
        model: Название модели
        messages: Список сообщений
        request_data: Данные запроса
        request_id: ID запроса для логирования
        
    Returns:
        dict: Подготовленный payload
    """
    # Форматируем историю диалога
    formatted_history = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        
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
            
    all_messages = "\n".join(formatted_history)
    
    # Проверяем запрос веб-поиска
    web_search = request_data.get("web_search", False)
    num_of_site = request_data.get("num_of_site", 3)
    max_word = request_data.get("max_word", 500)
    
    # Формируем payload
    payload = {
        "type": "CHAT_WITH_AI",
        "model": model,
        "promptObject": {
            "prompt": all_messages,
            "isMixed": False,
            "webSearch": web_search,
            "numOfSite": num_of_site if web_search else None,
            "maxWord": max_word if web_search else None,
        }
    }
    
    logger.debug(f"[{request_id}] Prepared chat payload for model {model}")
    return payload

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
