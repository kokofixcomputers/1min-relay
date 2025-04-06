# routes/audio.py

# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request
from utils.memcached import safe_memcached_operation
from . import app, limiter, MEMORY_STORAGE
from .utils import validate_auth, handle_api_error, format_openai_response

# Вспомогательные функции для устранения дублирования кода
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

def handle_error_response(error, api_key, request_id):
    """
    Обрабатывает ошибку и формирует соответствующий ответ
    
    Args:
        error: Объект ошибки (Exception или Response)
        api_key: API ключ
        request_id: ID запроса для логирования
        
    Returns:
        tuple: HTTP ответ с ошибкой
    """
    if isinstance(error, requests.Response):
        if error.status_code == 401:
            return ERROR_HANDLER(1020, key=api_key)
        
        logger.error(f"[{request_id}] API error: {error.text[:200] if hasattr(error, 'text') else str(error)}")
        error_text = "No available providers at the moment"
        try:
            error_json = error.json()
            if "error" in error_json:
                error_text = error_json["error"]
        except:
            pass
        
        return jsonify({"error": f"All available models failed. {error_text}"}), error.status_code
    else:
        logger.error(f"[{request_id}] Error: {str(error)}")
        return jsonify({"error": f"All available models failed. {str(error)}"}), 500


# Маршруты для работы с аудио
@app.route("/v1/audio/transcriptions", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_transcriptions():
    """
    Route for converting speech into text (analogue of Openai Whisper API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/audio/transcriptions")

    # Проверяем авторизацию
    api_key, error = validate_auth(request, request_id)
    if error:
        return error

    # Проверяем наличие аудио файла
    if "file" not in request.files:
        logger.error(f"[{request_id}] No audio file provided")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    model = request.form.get("model", "whisper-1")
    response_format = request.form.get("response_format", "text")
    language = request.form.get("language", None)
    temperature = request.form.get("temperature", 0)

    logger.info(f"[{request_id}] Processing audio transcription with model {model}")

    try:
        # Загружаем аудио файл
        audio_path, error = upload_audio_file(audio_file, api_key, request_id)
        if error:
            return error
        
        # Подготовка списка моделей для перебора
        models_to_try = prepare_models_list(model, SPEECH_TO_TEXT_MODELS)
        logger.debug(f"[{request_id}] Will try these models in order: {models_to_try}")
        
        # Функция для создания payload
        def create_transcription_payload(current_model):
            payload = {
                "type": "SPEECH_TO_TEXT",
                "model": current_model,
                "promptObject": {
                    "audioUrl": audio_path,
                    "response_format": response_format,
                },
            }
            
            # Добавляем дополнительные параметры если они предоставлены
            if language:
                payload["promptObject"]["language"] = language
                
            if temperature is not None:
                payload["promptObject"]["temperature"] = float(temperature)
                
            return payload
        
        # Пробуем модели по очереди
        one_min_response, error = try_models_in_sequence(
            models_to_try, create_transcription_payload, api_key, request_id
        )
        
        if error:
            return handle_error_response(error, api_key, request_id)
        
        # Извлекаем текст из ответа
        result_text = extract_text_from_response(one_min_response, request_id)
        
        if not result_text:
            logger.error(f"[{request_id}] Could not extract transcription text from API response")
            return jsonify({"error": "Could not extract transcription text"}), 500
        
        logger.info(f"[{request_id}] Successfully processed audio transcription")
        
        # Создаем json строго в формате Openai API
        response_data = {"text": result_text}
        response = jsonify(response_data)
        set_response_headers(response)
        return response

    except Exception as e:
        logger.error(f"[{request_id}] Exception during transcription request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/audio/translations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_translations():
    """
    Route for translating audio to text (analogue Openai Whisper API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/audio/translations")

    # Проверяем авторизацию
    api_key, error = validate_auth(request, request_id)
    if error:
        return error

    # Проверяем наличие аудио файла
    if "file" not in request.files:
        logger.error(f"[{request_id}] No audio file provided")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    model = request.form.get("model", "whisper-1")
    response_format = request.form.get("response_format", "text")
    temperature = request.form.get("temperature", 0)

    logger.info(f"[{request_id}] Processing audio translation with model {model}")

    try:
        # Загружаем аудио файл
        audio_path, error = upload_audio_file(audio_file, api_key, request_id)
        if error:
            return error
        
        # Подготовка списка моделей для перебора
        models_to_try = prepare_models_list(model, SPEECH_TO_TEXT_MODELS)
        logger.debug(f"[{request_id}] Will try these models in order: {models_to_try}")
        
        # Функция для создания payload
        def create_translation_payload(current_model):
            return {
                "type": "AUDIO_TRANSLATOR",
                "model": current_model,
                "promptObject": {
                    "audioUrl": audio_path,
                    "response_format": response_format,
                    "temperature": float(temperature),
                },
            }
        
        # Пробуем модели по очереди
        one_min_response, error = try_models_in_sequence(
            models_to_try, create_translation_payload, api_key, request_id
        )
        
        if error:
            return handle_error_response(error, api_key, request_id)
        
        # Извлекаем текст из ответа
        result_text = extract_text_from_response(one_min_response, request_id)
        
        if not result_text:
            logger.error(f"[{request_id}] Could not extract translation text from API response")
            return jsonify({"error": "Could not extract translation text"}), 500
        
        logger.info(f"[{request_id}] Successfully processed audio translation")
        
        # Создаем json строго в формате Openai API
        response_data = {"text": result_text}
        response = jsonify(response_data)
        set_response_headers(response)
        return response

    except Exception as e:
        logger.error(f"[{request_id}] Exception during translation request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/audio/speech", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def text_to_speech():
    """
    Route for converting text into speech (analogue Openai TTS API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = request.args.get('request_id', str(uuid.uuid4())[:8])
    logger.info(f"[{request_id}] Received request: /v1/audio/speech")

    # Проверяем авторизацию
    api_key, error = validate_auth(request, request_id)
    if error:
        return error

    # Получаем данные запроса
    request_data = {}
    
    # Проверяем наличие данных в Memcached если запрос был перенаправлен
    if 'request_id' in request.args and 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
        tts_session_key = f"tts_request_{request.args.get('request_id')}"
        try:
            session_data = safe_memcached_operation('get', tts_session_key)
            if session_data:
                if isinstance(session_data, str):
                    request_data = json.loads(session_data)
                elif isinstance(session_data, bytes):
                    request_data = json.loads(session_data.decode('utf-8'))
                else:
                    request_data = session_data
                    
                # Удаляем данные из кэша, они больше не нужны
                safe_memcached_operation('delete', tts_session_key)
                logger.debug(f"[{request_id}] Retrieved TTS request data from memcached")
        except Exception as e:
            logger.error(f"[{request_id}] Error retrieving TTS session data: {str(e)}")
    
    # Если данные не найдены в Memcache, пробуем получить их из тела запроса
    if not request_data and request.is_json:
        request_data = request.json
        
    model = request_data.get("model", "tts-1")
    input_text = request_data.get("input", "")
    voice = request_data.get("voice", "alloy")
    response_format = request_data.get("response_format", "mp3")
    speed = request_data.get("speed", 1.0)

    logger.info(f"[{request_id}] Processing TTS request with model {model}")
    logger.debug(f"[{request_id}] Text input: {input_text[:100]}..." if input_text and len(input_text) > 100 else f"[{request_id}] Text input: {input_text}")

    if not input_text:
        logger.error(f"[{request_id}] No input text provided")
        return jsonify({"error": "No input text provided"}), 400

    try:
        # Формируем Payload для TTS
        payload = {
            "type": "TEXT_TO_SPEECH",
            "model": model,
            "promptObject": {
                "text": input_text,
                "voice": voice,
                "response_format": response_format,
                "speed": speed
            }
        }

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}

        # Отправляем запрос
        logger.debug(f"[{request_id}] Sending TTS request to {ONE_MIN_API_URL}")
        response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
        logger.debug(f"[{request_id}] TTS response status code: {response.status_code}")

        if response.status_code != 200:
            return handle_api_error(response, api_key, request_id)

        # Обрабатываем ответ
        one_min_response = response.json()

        try:
            # Получаем URL аудио из ответа
            audio_url = extract_audio_url(one_min_response, request_id)
            
            if not audio_url:
                logger.error(f"[{request_id}] Could not extract audio URL from API response")
                return jsonify({"error": "Could not extract audio URL"}), 500

            # Получаем аудио данные по URL
            audio_response = api_request("GET", f"https://asset.1min.ai/{audio_url}")

            if audio_response.status_code != 200:
                logger.error(f"[{request_id}] Failed to download audio: {audio_response.status_code}")
                return jsonify({"error": "Failed to download audio"}), 500

            # Возвращаем аудио клиенту
            logger.info(f"[{request_id}] Successfully generated speech audio")

            # Создаем ответ с аудио и правильным MIME-type
            content_type = "audio/mpeg" if response_format == "mp3" else f"audio/{response_format}"
            response = make_response(audio_response.content)
            response.headers["Content-Type"] = content_type
            set_response_headers(response)

            return response

        except Exception as e:
            logger.error(f"[{request_id}] Error processing TTS response: {str(e)}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Exception during TTS request: {str(e)}")
        return jsonify({"error": str(e)}), 500


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
