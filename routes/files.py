# routes/files.py

# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request
from utils.memcached import safe_memcached_operation
from . import app, limiter
from .utils import validate_auth, handle_api_error, get_user_files, save_user_files, upload_asset, get_mime_type

# Вспомогательные функции для устранения дублирования кода
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

# Маршруты для работы с файлами
@app.route("/v1/files", methods=["GET", "POST", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_files():
    """
    Route for working with files: getting a list and downloading new files
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    
    # Проверяем авторизацию
    api_key, error = validate_auth(request, request_id)
    if error:
        return error

    # GET - получение списка файлов
    if request.method == "GET":
        logger.info(f"[{request_id}] Received request: GET /v1/files")
        try:
            # Получаем список файлов пользователя
            user_files = get_user_files(api_key, request_id)
            
            # Формируем ответ в формате OpenAI API
            files_data = []
            for file_info in user_files:
                if isinstance(file_info, dict) and "id" in file_info:
                    files_data.append(format_file_response(file_info))
                    
            response_data = {
                "data": files_data,
                "object": "list"
            }
            
            return create_api_response(response_data)
            
        except Exception as e:
            logger.error(f"[{request_id}] Exception during files list request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # POST - загрузка нового файла
    elif request.method == "POST":
        logger.info(f"[{request_id}] Received request: POST /v1/files")

        # Проверяем наличие файла
        if "file" not in request.files:
            logger.error(f"[{request_id}] No file provided")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        purpose = request.form.get("purpose", "assistants")

        try:
            # Получаем MIME тип файла
            mime_type, file_type = get_mime_type(file.filename)
            
            # Сохраняем содержимое файла, чтобы можно было получить и размер, и передать данные
            file_content = file.read()
            file_size = len(file_content)
            
            # Загружаем файл
            file_id, file_path, error = upload_asset(
                file_content,
                file.filename,
                mime_type,
                api_key,
                request_id,
                file_type
            )
            
            if error:
                return error
                
            # Получаем текущий список файлов пользователя
            user_files = get_user_files(api_key, request_id)
            
            # Добавляем новый файл в список
            file_info = {
                "id": file_id,
                "filename": file.filename,
                "bytes": file_size,
                "created_at": int(time.time())
            }
            user_files.append(file_info)
            
            # Сохраняем обновленный список
            save_user_files(api_key, user_files, request_id)
            
            # Формируем ответ
            response_data = format_file_response(file_info, purpose=purpose)

            logger.info(f"[{request_id}] File uploaded successfully: {file_id}")
            return create_api_response(response_data)

        except Exception as e:
            logger.error(f"[{request_id}] Exception during file upload: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route("/v1/files/<file_id>", methods=["GET", "DELETE", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file(file_id):
    """
    Route for working with a specific file: obtaining information and deleting
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    
    # Проверяем авторизацию
    api_key, error = validate_auth(request, request_id)
    if error:
        return error

    # GET - получение информации о файле
    if request.method == "GET":
        logger.info(f"[{request_id}] Received request: GET /v1/files/{file_id}")
        try:
            # Ищем файл в списке файлов пользователя
            user_files = get_user_files(api_key, request_id)
            file_info = find_file_by_id(user_files, file_id)
                    
            # Формируем ответ
            response_data = format_file_response(file_info, file_id)
            return create_api_response(response_data)

        except Exception as e:
            logger.error(f"[{request_id}] Exception during file info request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # DELETE - удаление файла
    elif request.method == "DELETE":
        logger.info(f"[{request_id}] Received request: DELETE /v1/files/{file_id}")
        try:
            # Получаем список файлов пользователя
            user_files = get_user_files(api_key, request_id)
            
            # Фильтруем список, исключая файл с указанным ID
            new_user_files = [f for f in user_files if f.get("id") != file_id]
            
            # Если список изменился, сохраняем обновленный список
            if len(new_user_files) < len(user_files):
                save_user_files(api_key, new_user_files, request_id)
                logger.info(f"[{request_id}] Deleted file {file_id} from user's files")

            # Возвращаем ответ об успешном удалении
            response_data = {
                "id": file_id,
                "object": "file",
                "deleted": True
            }

            return create_api_response(response_data)

        except Exception as e:
            logger.error(f"[{request_id}] Exception during file deletion: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route("/v1/files/<file_id>/content", methods=["GET", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file_content(file_id):
    """
    Route for obtaining the contents of the file
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: GET /v1/files/{file_id}/content")

    # Проверяем авторизацию
    api_key, error = validate_auth(request, request_id)
    if error:
        return error

    try:
        # В 1min.ai нет API для получения содержимого файла по ID
        logger.error(f"[{request_id}] File content retrieval not supported")
        return jsonify({"error": "File content retrieval not supported"}), 501

    except Exception as e:
        logger.error(f"[{request_id}] Exception during file content request: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
