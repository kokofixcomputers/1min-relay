# routes/functions/file_func.py

from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import (
    ERROR_HANDLER, 
    create_session, 
    api_request,
    set_response_headers
)
from utils.memcached import safe_memcached_operation
from flask import jsonify, make_response

#=======================================================#
# ----------- Функции для работы с файлами -------------#
#=======================================================#

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

def find_conversation_id(response_data, request_id=None):
    """
    Ищет ID разговора в ответе API
    
    Args:
        response_data: Данные ответа от API
        request_id: ID запроса для логирования
        
    Returns:
        str/None: ID разговора или None, если не найден
    """
    try:
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
    except Exception as e:
        logger.error(f"[{request_id}] Error finding conversation ID: {str(e)}")
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
