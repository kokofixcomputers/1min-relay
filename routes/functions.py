# routes/functions.py
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

# ----------- Общие функции для авторизации и обработки ошибок -----------

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

# ----------- Функции форматирования ответов -----------

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

# ----------- Функции для работы с файлами -----------

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

def format_file_response(file_info, file_id=None, purpose="assistants", status="processed"):
    """
    Форматирует информацию о файле в формат OpenAI API
    
    Args:
        file_info: Словарь с информацией о файле
        file_id: ID файла (если не указан в file_info)
        purpose: Назначение файла
        status: Статус обработки файла
        
    Returns:
        dict: Информация о файле в формате OpenAI API
    """
    # Если file_info не предоставлен, создаем пустой словарь
    if file_info is None:
        file_info = {}
    
    # Устанавливаем значения по умолчанию, если не указаны
    file_id = file_info.get("id", file_id)
    filename = file_info.get("filename", f"file_{file_id}")
    bytes_size = file_info.get("bytes", 0)
    created_at = file_info.get("created_at", int(time.time()))
    
    return {
        "id": file_id,
        "object": "file",
        "bytes": bytes_size,
        "created_at": created_at,
        "filename": filename,
        "purpose": purpose,
        "status": status
    }

def create_api_response(data, request_id=None):
    """
    Создает HTTP-ответ с правильными заголовками
    
    Args:
        data: Данные для ответа
        request_id: ID запроса для логирования
        
    Returns:
        Response: HTTP-ответ
    """
    response = make_response(jsonify(data))
    set_response_headers(response)
    return response

def find_conversation_id(response_data, request_id=None):
    """
    Ищет ID разговора в ответе API
    
    Args:
        response_data: Данные ответа от API
        request_id: ID запроса для логирования
        
    Returns:
        str/None: ID разговора или None, если не найден
    """
    # Сначала проверяем наиболее вероятные места
    if "conversation" in response_data and "uuid" in response_data["conversation"]:
        return response_data["conversation"]["uuid"]
    elif "id" in response_data:
        return response_data["id"]
    elif "uuid" in response_data:
        return response_data["uuid"]
    
    # Если не нашли, выполняем рекурсивный поиск
    def search_recursively(obj, path=""):
        if isinstance(obj, dict):
            if "id" in obj:
                logger.debug(f"[{request_id}] Found ID at path '{path}.id': {obj['id']}")
                return obj["id"]
            if "uuid" in obj:
                logger.debug(f"[{request_id}] Found UUID at path '{path}.uuid': {obj['uuid']}")
                return obj["uuid"]
                
            for key, value in obj.items():
                result = search_recursively(value, f"{path}.{key}")
                if result:
                    return result
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                result = search_recursively(item, f"{path}[{i}]")
                if result:
                    return result
        return None
    
    return search_recursively(response_data)

def find_file_by_id(user_files, file_id):
    """
    Находит файл в списке файлов пользователя по ID
    
    Args:
        user_files: Список файлов пользователя
        file_id: ID искомого файла
        
    Returns:
        dict/None: Информация о файле или None, если файл не найден
    """
    for file_item in user_files:
        if file_item.get("id") == file_id:
            return file_item
    return None

def create_conversation_with_files(file_ids, title, model, api_key, request_id=None):
    """
    Creates a new conversation with files
    
    Args:
        file_ids: List of file IDs
        title: The name of the conversation
        model: AI model
        api_key: API Key
        request_id: Request ID for logging
        
    Returns:
        str: Conversation ID or None in case of error
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Creating conversation with {len(file_ids)} files")
    
    try:
        # Формируем payload для запроса
        payload = {
            "title": title,
            "type": "CHAT_WITH_PDF",
            "model": model,
            "fileIds": file_ids,
        }
        
        logger.debug(f"[{request_id}] Conversation payload: {json.dumps(payload)}")
        
        # Используем правильный URL API
        conversation_url = "https://api.1min.ai/api/features/conversations?type=CHAT_WITH_PDF"
        
        logger.debug(f"[{request_id}] Creating conversation using URL: {conversation_url}")
        
        headers = {"API-KEY": api_key, "Content-Type": "application/json"}
        response = api_request("POST", conversation_url, json=payload, headers=headers)
        
        logger.debug(f"[{request_id}] Create conversation response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"[{request_id}] Failed to create conversation: {response.status_code} - {response.text}")
            return None
            
        response_data = response.json()
        logger.debug(f"[{request_id}] Conversation response data: {json.dumps(response_data)}")
        
        # Ищем ID разговора в разных местах ответа
        conversation_id = find_conversation_id(response_data, request_id)
            
        if not conversation_id:
            logger.error(f"[{request_id}] Could not find conversation ID in response: {response_data}")
            return None
            
        logger.info(f"[{request_id}] Conversation created successfully: {conversation_id}")
        return conversation_id
    except Exception as e:
        logger.error(f"[{request_id}] Error creating conversation: {str(e)}")
        return None

# ----------- Функции для работы с изображениями -----------
def get_full_url(url, asset_host="https://asset.1min.ai"):
    """Return full URL based on asset host."""
    if not url.startswith("http"):
        return f"{asset_host}{url}" if url.startswith("/") else f"{asset_host}/{url}"
    return url

def build_generation_payload(model, prompt, request_data, negative_prompt, aspect_ratio, size, mode, request_id):
    """Build payload for image generation based on model."""
    payload = {}
    if model == "dall-e-3":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dall-e-3",
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                "size": size or request_data.get("size", "1024x1024"),
                "quality": request_data.get("quality", "standard"),
                "style": request_data.get("style", "vivid"),
            },
        }
    elif model == "dall-e-2":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dall-e-2",
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                "size": size or request_data.get("size", "1024x1024"),
            },
        }
    elif model == "stable-diffusion-xl-1024-v1-0":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "stable-diffusion-xl-1024-v1-0",
            "promptObject": {
                "prompt": prompt,
                "samples": request_data.get("n", 1),
                "size": size or request_data.get("size", "1024x1024"),
                "cfg_scale": request_data.get("cfg_scale", 7),
                "clip_guidance_preset": request_data.get("clip_guidance_preset", "NONE"),
                "seed": request_data.get("seed", 0),
                "steps": request_data.get("steps", 30),
            },
        }
    elif model in ["midjourney", "midjourney_6_1"]:
        # Parse aspect ratio parts (default 1:1)
        try:
            ar_parts = tuple(map(int, aspect_ratio.split(":"))) if aspect_ratio else (1, 1)
        except Exception:
            ar_parts = (1, 1)
        model_name = "midjourney" if model == "midjourney" else "midjourney_6_1"
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model_name,
            "promptObject": {
                "prompt": prompt,
                "mode": mode or request_data.get("mode", "fast"),
                "n": 4,
                "aspect_width": ar_parts[0],
                "aspect_height": ar_parts[1],
                "isNiji6": request_data.get("isNiji6", False),
                "maintainModeration": request_data.get("maintainModeration", True),
                "image_weight": request_data.get("image_weight", 1),
                "weird": request_data.get("weird", 0),
            },
        }
        if negative_prompt or request_data.get("negativePrompt"):
            payload["promptObject"]["negativePrompt"] = negative_prompt or request_data.get("negativePrompt", "")
        if request_data.get("no", ""):
            payload["promptObject"]["no"] = request_data.get("no", "")
    elif model in ["black-forest-labs/flux-schnell", "flux-schnell",
                   "black-forest-labs/flux-dev", "flux-dev",
                   "black-forest-labs/flux-pro", "flux-pro",
                   "black-forest-labs/flux-1.1-pro", "flux-1.1-pro"]:
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model.split("/")[-1] if "/" in model else model,
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                "output_format": request_data.get("output_format", "webp"),
            },
        }
    elif model in [
        "6b645e3a-d64f-4341-a6d8-7a3690fbf042", "phoenix",
        "b24e16ff-06e3-43eb-8d33-4416c2d75876", "lightning-xl",
        "5c232a9e-9061-4777-980a-ddc8e65647c6", "vision-xl",
        "e71a1c2f-4f80-4800-934f-2c68979d8cc8", "anime-xl",
        "1e60896f-3c26-4296-8ecc-53e2afecc132", "diffusion-xl",
        "aa77f04e-3eec-4034-9c07-d0f619684628", "kino-xl",
        "2067ae52-33fd-4a82-bb92-c2c55e7d2786", "albedo-base-xl"
    ]:
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model,
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 4),
                "size": size,
                "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
            },
        }
        # Удаляем пустые параметры
        if not payload["promptObject"]["negativePrompt"]:
            del payload["promptObject"]["negativePrompt"]
        if model == "e71a1c2f-4f80-4800-934f-2c68979d8cc8":
            payload["promptObject"]["size"] = size or request_data.get("size", "1024x1024")
            payload["promptObject"]["aspect_ratio"] = aspect_ratio
            if not payload["promptObject"]["aspect_ratio"]:
                del payload["promptObject"]["aspect_ratio"]
    else:
        logger.error(f"[{request_id}] Invalid model: {model}")
        return None, ERROR_HANDLER(1002, model)
    return payload, None

def extract_image_urls_from_response(response_json, request_id):
    """Extract image URLs from API response."""
    image_urls = []
    result_object = response_json.get("aiRecord", {}).get("aiRecordDetail", {}).get("resultObject", [])
    if isinstance(result_object, list) and result_object:
        image_urls = result_object
    elif result_object and isinstance(result_object, str):
        image_urls = [result_object]
    if not image_urls and "resultObject" in response_json:
        result = response_json["resultObject"]
        if isinstance(result, list):
            image_urls = result
        elif isinstance(result, str):
            image_urls = [result]
    if not image_urls:
        logger.error(f"[{request_id}] Could not extract image URLs from API response: {json.dumps(response_json)[:500]}")
    return image_urls

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

def parse_aspect_ratio(prompt, model, request_data, request_id=None):
    """
    Parse aspect ratio, size and other parameters from the prompt.
    Enhanced version combining functionality from both implementations.
    Returns: (modified prompt, aspect_ratio, size, error_message, mode)
    """
    original_prompt = prompt
    mode = None
    size = request_data.get("size", "1024x1024")
    aspect_ratio = None
    ar_error = None

    try:
        # Extract mode parameter (--fast or --relax)
        mode_match = re.search(r'(--|\u2014)(fast|relax)\s*', prompt)
        if mode_match:
            mode = mode_match.group(2)
            prompt = re.sub(r'(--|\u2014)(fast|relax)\s*', '', prompt).strip()
            logger.debug(f"[{request_id}] Extracted mode: {mode}")

        # Extract size parameter
        size_match = re.search(r'(--|\u2014)size\s+(\d+x\d+)', prompt)
        if size_match:
            size = size_match.group(2)
            prompt = re.sub(r'(--|\u2014)size\s+\d+x\d+\s*', '', prompt).strip()
            logger.debug(f"[{request_id}] Extracted size: {size}")

        # Extract aspect ratio from prompt
        ar_match = re.search(r'(--|\u2014)ar\s+(\d+):(\d+)', prompt)
        if ar_match:
            width = int(ar_match.group(2))
            height = int(ar_match.group(3))
            
            # Validate aspect ratio
            if width <= 0 or height <= 0:
                logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height}")
                return original_prompt, None, size, "Aspect ratio dimensions must be positive", mode
            
            # Check aspect ratio limits
            if max(width, height) / min(width, height) > 2:
                ar_error = "Aspect ratio cannot exceed 2:1 or 1:2"
                logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height} - {ar_error}")
                return prompt, None, size, ar_error, mode
                
            if width > 10000 or height > 10000:
                ar_error = "Aspect ratio values must be between 1 and 10000"
                logger.error(f"[{request_id}] Invalid aspect ratio values: {width}:{height} - {ar_error}")
                return prompt, None, size, ar_error, mode
            
            # Simplify aspect ratio if needed
            if width > 10 or height > 10:
                gcd_val = math.gcd(width, height)
                width = width // gcd_val
                height = height // gcd_val
            
            aspect_ratio = f"{width}:{height}"
            prompt = re.sub(r'(--|\u2014)ar\s+\d+:\d+\s*', '', prompt).strip()
            logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_ratio}")
        # Check for aspect ratio in request data
        elif "aspect_ratio" in request_data:
            aspect_ratio = request_data.get("aspect_ratio")
            if not re.match(r'^\d+:\d+$', aspect_ratio):
                ar_error = "Aspect ratio must be in format width:height"
                logger.error(f"[{request_id}] Invalid aspect ratio format: {aspect_ratio} - {ar_error}")
                return prompt, None, size, ar_error, mode
            width, height = map(int, aspect_ratio.split(':'))
            if max(width, height) / min(width, height) > 2:
                ar_error = "Aspect ratio cannot exceed 2:1 or 1:2"
                logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height} - {ar_error}")
                return prompt, None, size, ar_error, mode
            if width < 1 or width > 10000 or height < 1 or height > 10000:
                ar_error = "Aspect ratio values must be between 1 and 10000"
                logger.error(f"[{request_id}] Invalid aspect ratio values: {width}:{height} - {ar_error}")
                return prompt, None, size, ar_error, mode
            logger.debug(f"[{request_id}] Using aspect ratio from request: {aspect_ratio}")
            
        # Remove negative prompt parameters
        prompt = re.sub(r'(--|\u2014)no\s+.*?(?=(--|\u2014)|$)', '', prompt).strip()
            
        # Handle special case for dall-e-3 which doesn't support custom aspect ratio
        if model == "dall-e-3" and aspect_ratio:
            width, height = map(int, aspect_ratio.split(':'))
            if abs(width / height - 1) < 0.1:
                size = "1024x1024"
                aspect_ratio = "square"
            elif width > height:
                size = "1792x1024"
                aspect_ratio = "landscape"
            else:
                size = "1024x1792"
                aspect_ratio = "portrait"
            logger.debug(f"[{request_id}] Adjusted size for DALL-E 3: {size}, aspect_ratio: {aspect_ratio}")
        # Special adjustments for Leonardo models
        elif model in [
            "6b645e3a-d64f-4341-a6d8-7a3690fbf042", "phoenix",
            "b24e16ff-06e3-43eb-8d33-4416c2d75876", "lightning-xl",
            "5c232a9e-9061-4777-980a-ddc8e65647c6", "vision-xl",
            "e71a1c2f-4f80-4800-934f-2c68979d8cc8", "anime-xl",
            "1e60896f-3c26-4296-8ecc-53e2afecc132", "diffusion-xl",
            "aa77f04e-3eec-4034-9c07-d0f619684628", "kino-xl",
            "2067ae52-33fd-4a82-bb92-c2c55e7d2786", "albedo-base-xl"
        ] and aspect_ratio:
            if aspect_ratio == "1:1":
                size = LEONARDO_SIZES["1:1"]
            elif aspect_ratio == "4:3":
                size = LEONARDO_SIZES["4:3"]
            elif aspect_ratio == "3:4":
                size = LEONARDO_SIZES["3:4"]
            else:
                width, height = map(int, aspect_ratio.split(':'))
                ratio = width / height
                if abs(ratio - 1) < 0.1:
                    size = LEONARDO_SIZES["1:1"]
                    aspect_ratio = "1:1"
                elif ratio > 1:
                    size = LEONARDO_SIZES["4:3"]
                    aspect_ratio = "4:3"
                else:
                    size = LEONARDO_SIZES["3:4"]
                    aspect_ratio = "3:4"
            logger.debug(f"[{request_id}] Adjusted size for Leonardo model: {size}, aspect_ratio: {aspect_ratio}")
        
        return prompt, aspect_ratio, size, ar_error, mode
    
    except Exception as e:
        logger.error(f"[{request_id}] Error parsing aspect ratio: {str(e)}")
        return original_prompt, None, size, f"Error parsing parameters: {str(e)}", mode

def retry_image_upload(image_url, api_key, request_id=None):
    """
    Attempts to re-upload an existing image to the asset server.
    
    Args:
        image_url: URL of the existing image
        api_key: API key for authorization
        request_id: Request ID for logging
        
    Returns:
        tuple: (new_image_url, error_response)
    """
    try:
        # Download the image from the URL
        session = create_session()
        try:
            if not image_url.startswith("http"):
                if image_url.startswith("/"):
                    image_url = f"https://asset.1min.ai{image_url}"
                else:
                    image_url = f"https://asset.1min.ai/{image_url}"
                    
            logger.debug(f"[{request_id}] Downloading image from {image_url}")
            response = session.get(image_url, timeout=DOWNLOAD_TIMEOUT)
            
            if response.status_code != 200:
                logger.error(f"[{request_id}] Failed to download image: {response.status_code}")
                return None, (jsonify({"error": "Failed to download image for re-upload"}), response.status_code)
                
            image_data = response.content
            logger.debug(f"[{request_id}] Successfully downloaded image ({len(image_data)} bytes)")
            
            # Determine file extension from the URL or content type
            file_extension = os.path.splitext(image_url)[1]
            if not file_extension:
                content_type = response.headers.get("Content-Type", "")
                if "jpeg" in content_type or "jpg" in content_type:
                    file_extension = ".jpg"
                elif "png" in content_type:
                    file_extension = ".png"
                elif "webp" in content_type:
                    file_extension = ".webp"
                else:
                    file_extension = ".jpg"  # Default to jpg
                    
            # Upload image back to 1min.ai
            filename = f"reupload{file_extension}"
            mime_type = get_mime_type(filename)[0]
            
            asset_id, asset_path, asset_error = upload_asset(
                image_data, filename, mime_type, api_key, request_id
            )
            
            if asset_error:
                return None, asset_error
                
            if not asset_path:
                logger.error(f"[{request_id}] Failed to get asset path after re-upload")
                return None, (jsonify({"error": "Failed to re-upload image"}), 500)
                
            # Format the new URL
            new_url = f"https://asset.1min.ai{asset_path}" if asset_path.startswith("/") else f"https://asset.1min.ai/{asset_path}"
            logger.debug(f"[{request_id}] Successfully re-uploaded image: {new_url}")
            
            return new_url, None
        finally:
            session.close()
    except Exception as e:
        logger.error(f"[{request_id}] Error during image re-upload: {str(e)}")
        return None, (jsonify({"error": f"Failed to re-upload image: {str(e)}"}), 500)

def create_image_variations(image_url, user_model, n, aspect_width=None, aspect_height=None, mode=None, request_id=None):
    """
    Generate variations of the uploaded image using the 1min.ai API.
    Enhanced version combining functionality from both implementations.
    
    Args:
        image_url: URL of the uploaded image
        user_model: Requested model name
        n: Number of variations to generate
        aspect_width: Width for aspect ratio (optional)
        aspect_height: Height for aspect ratio (optional)
        mode: Generation mode (optional)
        request_id: Request ID for logging
        
    Returns:
        list: Image URLs of the generated variations or tuple (response, status_code) in case of error
    """
    # Set request_id if not provided
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
        
    variation_urls = []
    current_model = None
    
    # Try to get saved generation parameters from memcached
    generation_params = None
    try:
        gen_key = f"gen_params:{request_id}"
        params_json = safe_memcached_operation('get', gen_key)
        if params_json:
            if isinstance(params_json, str):
                generation_params = json.loads(params_json)
            elif isinstance(params_json, bytes):
                generation_params = json.loads(params_json.decode('utf-8'))
            logger.debug(f"[{request_id}] Retrieved generation parameters from memcached: {generation_params}")
    except Exception as e:
        logger.error(f"[{request_id}] Error retrieving generation parameters: {str(e)}")
        
    # Use saved parameters if available
    if generation_params:
        if "aspect_width" in generation_params and "aspect_height" in generation_params:
            aspect_width = generation_params.get("aspect_width")
            aspect_height = generation_params.get("aspect_height")
            logger.debug(f"[{request_id}] Using saved aspect ratio: {aspect_width}:{aspect_height}")
        if "mode" in generation_params:
            mode = generation_params.get("mode")
            logger.debug(f"[{request_id}] Using saved mode: {mode}")
    
    # Determine which models to try for variations
    variation_models = []
    if user_model in VARIATION_SUPPORTED_MODELS:
        variation_models.append(user_model)
    # Add fallback models
    variation_models.extend([m for m in ["midjourney_6_1", "midjourney", "clipdrop", "dall-e-2"] if m != user_model])
    variation_models = list(dict.fromkeys(variation_models))
    logger.info(f"[{request_id}] Trying image variations with models: {variation_models}")
    
    try:
        # Get API key from request
        auth_header = request.headers.get("Authorization", "")
        api_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        
        if not api_key:
            logger.error(f"[{request_id}] No API key provided for variation")
            return None
            
        headers = {"API-KEY": api_key, "Content-Type": "application/json"}
        session = create_session()
        
        try:
            # Download the image from the URL
            image_response = session.get(image_url, stream=True, timeout=MIDJOURNEY_TIMEOUT)
            if image_response.status_code != 200:
                logger.error(f"[{request_id}] Failed to download image: {image_response.status_code}")
                return jsonify({"error": "Failed to download image"}), 500
                
            # Try each model in sequence
            for model in variation_models:
                current_model = model
                logger.info(f"[{request_id}] Trying model: {model} for image variations")
                
                try:
                    # Determine MIME type and extension
                    content_type = "image/png"
                    if "content-type" in image_response.headers:
                        content_type = image_response.headers["content-type"]
                    elif image_url.lower().endswith(".webp"):
                        content_type = "image/webp"
                    elif image_url.lower().endswith(".jpg") or image_url.lower().endswith(".jpeg"):
                        content_type = "image/jpeg"
                    elif image_url.lower().endswith(".gif"):
                        content_type = "image/gif"
                    
                    ext = "png"
                    if "webp" in content_type:
                        ext = "webp"
                    elif "jpeg" in content_type or "jpg" in content_type:
                        ext = "jpg"
                    elif "gif" in content_type:
                        ext = "gif"
                    logger.debug(f"[{request_id}] Detected image type: {content_type}, extension: {ext}")
                    
                    # Upload image to server
                    files = {"asset": (f"variation.{ext}", image_response.content, content_type)}
                    upload_response = session.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
                    
                    if upload_response.status_code != 200:
                        logger.error(f"[{request_id}] Image upload failed: {upload_response.status_code}")
                        continue
                        
                    upload_data = upload_response.json()
                    logger.debug(f"[{request_id}] Asset upload response: {upload_data}")
                    
                    image_path = None
                    if "fileContent" in upload_data and "path" in upload_data["fileContent"]:
                        image_path = upload_data["fileContent"]["path"]
                        if image_path.startswith('/'):
                            image_path = image_path[1:]
                        logger.debug(f"[{request_id}] Using relative path for variation: {image_path}")
                    else:
                        logger.error(f"[{request_id}] Could not extract image path from upload response")
                        continue
                    
                    # Create model-specific payload
                    payload = {}
                    if model in ["midjourney_6_1", "midjourney"]:
                        payload = {
                            "type": "IMAGE_VARIATOR",
                            "model": model,
                            "promptObject": {
                                "imageUrl": image_path,
                                "mode": mode or "fast",
                                "n": 4,
                                "isNiji6": False,
                                "aspect_width": aspect_width or 1,
                                "aspect_height": aspect_height or 1,
                                "maintainModeration": True
                            }
                        }
                        logger.info(f"[{request_id}] Midjourney variation payload: {json.dumps(payload['promptObject'], indent=2)}")
                    elif model == "dall-e-2":
                        payload = {
                            "type": "IMAGE_VARIATOR",
                            "model": "dall-e-2",
                            "promptObject": {
                                "imageUrl": image_path,
                                "n": 1,
                                "size": "1024x1024"
                            }
                        }
                        logger.info(f"[{request_id}] DALL-E 2 variation payload: {json.dumps(payload, indent=2)}")
                        
                        # Try DALL-E 2 specific endpoint first
                        variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=MIDJOURNEY_TIMEOUT)
                        if variation_response.status_code != 200:
                            logger.error(f"[{request_id}] DALL-E 2 variation failed: {variation_response.status_code}, {variation_response.text}")
                            continue
                            
                        variation_data = variation_response.json()
                        if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                            result_object = variation_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
                            if isinstance(result_object, list):
                                variation_urls.extend(result_object)
                            elif isinstance(result_object, str):
                                variation_urls.append(result_object)
                        elif "resultObject" in variation_data:
                            result_object = variation_data["resultObject"]
                            if isinstance(result_object, list):
                                variation_urls.extend(result_object)
                            elif isinstance(result_object, str):
                                variation_urls.append(result_object)
                                
                        if variation_urls:
                            logger.info(f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2")
                            break
                        else:
                            logger.warning(f"[{request_id}] No variation URLs found in DALL-E 2 response")
                    elif model == "clipdrop":
                        payload = {
                            "type": "IMAGE_VARIATOR",
                            "model": "clipdrop",
                            "promptObject": {
                                "imageUrl": image_path,
                                "n": n
                            }
                        }
                        logger.info(f"[{request_id}] Clipdrop variation payload: {json.dumps(payload, indent=2)}")
                        
                        # Try Clipdrop specific endpoint
                        variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=MIDJOURNEY_TIMEOUT)
                        if variation_response.status_code != 200:
                            logger.error(f"[{request_id}] Clipdrop variation failed: {variation_response.status_code}, {variation_response.text}")
                            continue
                            
                        variation_data = variation_response.json()
                        if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                            result_object = variation_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
                            if isinstance(result_object, list):
                                variation_urls.extend(result_object)
                            elif isinstance(result_object, str):
                                variation_urls.append(result_object)
                        elif "resultObject" in variation_data:
                            result_object = variation_data["resultObject"]
                            if isinstance(result_object, list):
                                variation_urls.extend(result_object)
                            elif isinstance(result_object, str):
                                variation_urls.append(result_object)
                                
                        if variation_urls:
                            logger.info(f"[{request_id}] Successfully created {len(variation_urls)} variations with Clipdrop")
                            break
                        else:
                            logger.warning(f"[{request_id}] No variation URLs found in Clipdrop response")
                    
                    # If we reach here for midjourney or if previous attempts didn't succeed, try main API endpoint
                    if payload:
                        timeout = MIDJOURNEY_TIMEOUT if model.startswith("midjourney") else DEFAULT_TIMEOUT
                        
                        # Make the API request
                        variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=timeout)
                        
                        if variation_response.status_code != 200:
                            logger.error(f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
                            # When the Gateway Timeout (504) error, we return the error immediately, and do not continue to process
                            if variation_response.status_code == 504:
                                logger.error(f"[{request_id}] Midjourney API timeout (504). Returning error to client instead of fallback.")
                                return jsonify({
                                    "error": "Gateway Timeout (504) occurred while processing image variation request. Try again later."
                                }), 504
                            continue
                        
                        # Process the response
                        variation_data = variation_response.json()
                        
                        # Extract variation URLs from response
                        if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                            result_object = variation_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
                            if isinstance(result_object, list):
                                variation_urls.extend(result_object)
                            elif isinstance(result_object, str):
                                variation_urls.append(result_object)
                        elif "resultObject" in variation_data:
                            result_object = variation_data["resultObject"]
                            if isinstance(result_object, list):
                                variation_urls.extend(result_object)
                            elif isinstance(result_object, str):
                                variation_urls.append(result_object)
                        
                        if variation_urls:
                            logger.info(f"[{request_id}] Successfully created {len(variation_urls)} variations with {model}")
                            break
                        else:
                            logger.warning(f"[{request_id}] No variation URLs found in response for model {model}")
                
                except Exception as e:
                    logger.error(f"[{request_id}] Error with model {model}: {str(e)}")
                    continue
            
            # Handle case where all models failed
            if not variation_urls:
                logger.error(f"[{request_id}] Failed to create variations with any available model")
                return jsonify({"error": "Failed to create image variations with any available model"}), 500
            
            # Format the successful response
            logger.info(f"[{request_id}] Generated {len(variation_urls)} image variations with {current_model}")
            return variation_urls
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"[{request_id}] Error generating image variations: {str(e)}")
        return jsonify({"error": str(e)}), 500

def retry_image_upload(image_url, api_key, request_id=None):
    """Uploads an image with repeated attempts, returns a direct link to it."""
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Uploading image: {image_url}")
    session = create_session()
    temp_file_path = None
    try:
        if image_url.startswith(("http://", "https://")):
            logger.debug(f"[{request_id}] Fetching image from URL: {image_url}")
            response = session.get(image_url, stream=True)
            response.raise_for_status()
            image_data = response.content
        else:
            logger.debug(f"[{request_id}] Decoding base64 image")
            image_data = base64.b64decode(image_url.split(",")[1])
        if len(image_data) == 0:
            logger.error(f"[{request_id}] Empty image data")
            return None
        temp_file_path = safe_temp_file("image", request_id)
        with open(temp_file_path, "wb") as f:
            f.write(image_data)
        if os.path.getsize(temp_file_path) == 0:
            logger.error(f"[{request_id}] Empty image file created: {temp_file_path}")
            return None
        try:
            with open(temp_file_path, "rb") as f:
                upload_response = session.post(
                    ONE_MIN_ASSET_URL,
                    headers={"API-KEY": api_key},
                    files={"asset": (os.path.basename(image_url),
                                     f,
                                     "image/webp" if image_url.endswith(".webp") else "image/jpeg")}
                )
                if upload_response.status_code != 200:
                    logger.error(f"[{request_id}] Upload failed with status {upload_response.status_code}: {upload_response.text}")
                    return None
                upload_data = upload_response.json()
                if isinstance(upload_data, str):
                    try:
                        upload_data = json.loads(upload_data)
                    except:
                        logger.error(f"[{request_id}] Failed to parse upload response: {upload_data}")
                        return None
                logger.debug(f"[{request_id}] Upload response: {upload_data}")
                if "fileContent" in upload_data and "path" in upload_data["fileContent"]:
                    url = upload_data["fileContent"]["path"]
                    logger.info(f"[{request_id}] Image uploaded successfully: {url}")
                    return url
                logger.error(f"[{request_id}] No path found in upload response")
                return None
        except Exception as e:
            logger.error(f"[{request_id}] Exception during image upload: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"[{request_id}] Exception during image processing: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        session.close()
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"[{request_id}] Removed temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to remove temp file {temp_file_path}: {str(e)}")

# ----------- Функции для работы с текстовыми моделями -----------

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

    # Add new input if it is
    if new_input:
        formatted_history.append(f"User: {new_input}")

    # We return only the history of dialogue without additional instructions
    return "\n".join(formatted_history)

def get_model_capabilities(model):
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

    # We check the support of each opportunity through the corresponding arrays
    capabilities["vision"] = model in vision_supported_models
    capabilities["code_interpreter"] = model in code_interpreter_supported_models
    capabilities["retrieval"] = model in retrieval_supported_models
    capabilities["function_calling"] = model in function_calling_supported_models

    return capabilities

def prepare_payload(
        request_data, model, all_messages, image_paths=None, request_id=None
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
    capabilities = get_model_capabilities(model)

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

    # We form the basic Payload
    if image_paths:
        # Even if the model does not support images, we try to send as a text request
        if capabilities["vision"]:
            # Add instructions to the prompt field
            enhanced_prompt = all_messages
            if not enhanced_prompt.strip().startswith(IMAGE_DESCRIPTION_INSTRUCTION):
                enhanced_prompt = f"{IMAGE_DESCRIPTION_INSTRUCTION}\n\n{all_messages}"

            payload = {
                "type": "CHAT_WITH_IMAGE",
                "model": model,
                "promptObject": {
                    "prompt": enhanced_prompt,
                    "isMixed": False,
                    "imageList": image_paths,
                    "webSearch": web_search,
                    "numOfSite": num_of_site if web_search else None,
                    "maxWord": max_word if web_search else None,
                },
            }

            if web_search:
                logger.debug(
                    f"[{request_id}] Web search enabled in payload with numOfSite={num_of_site}, maxWord={max_word}")
        else:
            logger.debug(
                f"[{request_id}] Model {model} does not support vision, falling back to text-only chat"
            )
            payload = {
                "type": "CHAT_WITH_AI",
                "model": model,
                "promptObject": {
                    "prompt": all_messages,
                    "isMixed": False,
                    "webSearch": web_search,
                    "numOfSite": num_of_site if web_search else None,
                    "maxWord": max_word if web_search else None,
                },
            }

            if web_search:
                logger.debug(
                    f"[{request_id}] Web search enabled in payload with numOfSite={num_of_site}, maxWord={max_word}")
    elif code_interpreter:
        # If Code_interpreter is requested and supported
        payload = {
            "type": "CODE_GENERATOR",
            "model": model,
            "conversationId": "CODE_GENERATOR",
            "promptObject": {"prompt": all_messages},
        }
    else:
        # Basic text request
        payload = {
            "type": "CHAT_WITH_AI",
            "model": model,
            "promptObject": {
                "prompt": all_messages,
                "isMixed": False,
                "webSearch": web_search,
                "numOfSite": num_of_site if web_search else None,
                "maxWord": max_word if web_search else None,
            },
        }

        if web_search:
            logger.debug(
                f"[{request_id}] Web search enabled in payload with numOfSite={num_of_site}, maxWord={max_word}")

    return payload

def transform_response(one_min_response, request_data, prompt_token):
    try:
        # Output of the response structure for debugging
        logger.debug(f"Response structure: {json.dumps(one_min_response)[:200]}...")

        # We get an answer from the appropriate place to json
        result_text = (
            one_min_response.get("aiRecord", {})
            .get("aiRecordDetail", {})
            .get("resultObject", [""])[0]
        )

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

        completion_token = calculate_token(result_text)
        logger.debug(
            f"Finished processing Non-Streaming response. Completion tokens: {str(completion_token)}"
        )
        logger.debug(f"Total tokens: {str(completion_token + prompt_token)}")

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
                        "content": result_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token,
                "completion_tokens": completion_token,
                "total_tokens": prompt_token + completion_token,
            },
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

# ----------- Функции для работы с аудио -----------

def upload_audio_file(audio_file, api_key, request_id):
    """
    Загружает аудио файл в 1min.ai
    
    Args:
        audio_file: Файл аудио
        api_key: API ключ
        request_id: ID запроса для логирования
        
    Returns:
        tuple: (audio_path, error_response)
        audio_path будет None если произошла ошибка
    """
    try:
        session = create_session()
        headers = {"API-KEY": api_key}
        files = {"asset": (audio_file.filename, audio_file, "audio/mpeg")}
        
        try:
            asset_response = session.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
            logger.debug(f"[{request_id}] Audio upload response status code: {asset_response.status_code}")
            
            if asset_response.status_code != 200:
                error_message = asset_response.json().get("error", "Failed to upload audio")
                return None, (jsonify({"error": error_message}), asset_response.status_code)
                
            audio_path = asset_response.json()["fileContent"]["path"]
            logger.debug(f"[{request_id}] Successfully uploaded audio: {audio_path}")
            return audio_path, None
        finally:
            session.close()
    except Exception as e:
        logger.error(f"[{request_id}] Error uploading audio: {str(e)}")
        return None, (jsonify({"error": f"Failed to upload audio: {str(e)}"}), 500)

def try_models_in_sequence(models_to_try, payload_func, api_key, request_id):
    """
    Пробует использовать модели по очереди, пока одна не сработает
    
    Args:
        models_to_try: Список моделей для перебора
        payload_func: Функция создания payload для каждой модели
        api_key: API ключ
        request_id: ID запроса для логирования
        
    Returns:
        tuple: (result, error)
        result будет None если все модели завершились с ошибкой
    """
    last_error = None
    
    for current_model in models_to_try:
        try:
            payload = payload_func(current_model)
            headers = {"API-KEY": api_key, "Content-Type": "application/json"}
            
            logger.debug(f"[{request_id}] Trying model {current_model}")
            response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
            logger.debug(f"[{request_id}] Response status code: {response.status_code} for model {current_model}")
            
            if response.status_code == 200:
                one_min_response = response.json()
                return one_min_response, None
            else:
                # Сохраняем ошибку и пробуем следующую модель
                last_error = response
                logger.warning(f"[{request_id}] Model {current_model} failed with status {response.status_code}")
        
        except Exception as e:
            logger.error(f"[{request_id}] Error with model {current_model}: {str(e)}")
            last_error = e
    
    # Если мы дошли до сюда, значит ни одна модель не сработала
    logger.error(f"[{request_id}] All available models failed")
    
    # Возвращаем последнюю ошибку
    return None, last_error

def extract_text_from_response(response_data, request_id):
    """
    Извлекает текст из ответа API
    
    Args:
        response_data: Данные ответа от API
        request_id: ID запроса для логирования
        
    Returns:
        str: Извлеченный текст или пустая строка в случае ошибки
    """
    result_text = ""
    
    if "aiRecord" in response_data and "aiRecordDetail" in response_data["aiRecord"]:
        result_text = response_data["aiRecord"]["aiRecordDetail"].get("resultObject", [""])[0]
    elif "resultObject" in response_data:
        result_text = (
            response_data["resultObject"][0]
            if isinstance(response_data["resultObject"], list)
            else response_data["resultObject"]
        )
    
    # Проверяем если result_text это json
    try:
        if result_text and isinstance(result_text, str) and result_text.strip().startswith("{"):
            parsed_json = json.loads(result_text)
            if "text" in parsed_json:
                result_text = parsed_json["text"]
                logger.debug(f"[{request_id}] Extracted inner text from JSON")
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    
    if not result_text:
        logger.error(f"[{request_id}] Could not extract text from API response")
        
    return result_text

def prepare_models_list(requested_model, available_models):
    """
    Подготавливает список моделей для последовательного перебора
    
    Args:
        requested_model: Запрошенная модель
        available_models: Список доступных моделей
        
    Returns:
        list: Список моделей для перебора
    """
    if requested_model in available_models:
        # Если запрошена конкретная модель, пробуем её первой
        models_to_try = [requested_model]
        # Добавляем остальные модели из списка, кроме уже добавленной
        models_to_try.extend([m for m in available_models if m != requested_model])
    else:
        # Если запрошенная модель не в списке, используем все модели из списка
        models_to_try = available_models
        
    return models_to_try

def get_audio_from_url(audio_url, request_id):
    """
    Получает аудио данные по URL
    
    Args:
        audio_url: URL аудио файла
        request_id: ID запроса для логирования
        
    Returns:
        tuple: (audio_data, content_type, error_response)
        audio_data будет None если произошла ошибка
    """
    try:
        full_url = f"https://asset.1min.ai/{audio_url}"
        audio_response = api_request("GET", full_url)
        
        if audio_response.status_code != 200:
            logger.error(f"[{request_id}] Failed to download audio: {audio_response.status_code}")
            return None, None, (jsonify({"error": "Failed to download audio"}), 500)
        
        logger.info(f"[{request_id}] Successfully downloaded audio data")
        return audio_response.content, None
    except Exception as e:
        logger.error(f"[{request_id}] Error downloading audio: {str(e)}")
        return None, (jsonify({"error": f"Failed to download audio: {str(e)}"}), 500)

def extract_audio_url(response_data, request_id):
    """
    Извлекает URL аудио из ответа API
    
    Args:
        response_data: Данные ответа от API
        request_id: ID запроса для логирования
        
    Returns:
        str: URL аудио или пустая строка в случае ошибки
    """
    audio_url = ""
    
    if "aiRecord" in response_data and "aiRecordDetail" in response_data["aiRecord"]:
        result_object = response_data["aiRecord"]["aiRecordDetail"].get("resultObject", "")
        if isinstance(result_object, list) and result_object:
            audio_url = result_object[0]
        else:
            audio_url = result_object
    elif "resultObject" in response_data:
        result_object = response_data["resultObject"]
        if isinstance(result_object, list) and result_object:
            audio_url = result_object[0]
        else:
            audio_url = result_object
    
    if not audio_url:
        logger.error(f"[{request_id}] Could not extract audio URL from API response")
        
    return audio_url
