# routes/audio.py

# Импортируем только необходимые модули
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
from routes.functions.shared_func import validate_auth, handle_api_error, extract_text_from_response, extract_audio_url
from routes.functions.audio_func import upload_audio_file, try_models_in_sequence, prepare_models_list, prepare_whisper_payload, prepare_tts_payload
from . import app, limiter, MEMORY_STORAGE

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
