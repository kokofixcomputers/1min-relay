# routes/files.py

# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, api_request
from utils.memcached import safe_memcached_operation
from . import app, limiter
from .functions import (
    validate_auth, 
    handle_api_error, 
    get_user_files, 
    save_user_files, 
    upload_asset, 
    get_mime_type,
    format_file_response,
    create_api_response,
    find_file_by_id
)

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
            
            if not file_info:
                logger.error(f"[{request_id}] File {file_id} not found")
                return jsonify({"error": f"File {file_id} not found"}), 404
                    
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
            
            # Проверяем, существует ли файл
            found = False
            for file in user_files:
                if file.get("id") == file_id:
                    found = True
                    break
                    
            if not found:
                logger.error(f"[{request_id}] File {file_id} not found")
                return jsonify({"error": f"File {file_id} not found"}), 404
            
            # Фильтруем список, исключая файл с указанным ID
            new_user_files = [f for f in user_files if f.get("id") != file_id]
            
            # Сохраняем обновленный список
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
        # Проверяем, существует ли файл
        user_files = get_user_files(api_key, request_id)
        file_info = find_file_by_id(user_files, file_id)
        
        if not file_info:
            logger.error(f"[{request_id}] File {file_id} not found")
            return jsonify({"error": f"File {file_id} not found"}), 404
            
        # В 1min.ai нет API для получения содержимого файла по ID
        logger.error(f"[{request_id}] File content retrieval not supported")
        return jsonify({"error": "File content retrieval not supported"}), 501

    except Exception as e:
        logger.error(f"[{request_id}] Exception during file content request: {str(e)}")
        return jsonify({"error": str(e)}), 500

