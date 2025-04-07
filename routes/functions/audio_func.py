# routes/functions/audio_func.py

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
from routes.functions.shared_func import extract_text_from_response, extract_audio_url

#=======================================================#
# ----------- Функции для работы с аудио ---------------
#=======================================================#

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

def prepare_models_list(requested_model, available_models):
    """
    Подготавливает список моделей для обработки
    
    Args:
        requested_model: Запрошенная модель
        available_models: Доступные модели
        
    Returns:
        list: Список моделей для обработки
    """
    # Проверяем наличие модели в списке доступных
    if requested_model in available_models:
        # Если модель есть в списке, пробуем её первой
        models = [requested_model] + [m for m in available_models if m != requested_model]
    else:
        # Если модели нет, используем все доступные
        models = available_models
        
    return models

def prepare_whisper_payload(model, file_path, language=None, prompt=None, temperature=None, response_format=None):
    """
    Подготавливает данные для запроса к API транскрипции аудио
    
    Args:
        model: Модель для транскрипции
        file_path: Путь к аудиофайлу
        language: Язык аудио (опционально)
        prompt: Подсказка для транскрипции (опционально)
        temperature: Температура генерации (опционально)
        response_format: Формат ответа (опционально)
        
    Returns:
        dict: Данные для запроса
    """
    payload = {
        "model": "whisper_v2",
        "file": (os.path.basename(file_path), open(file_path, "rb"), "audio/mpeg")
    }
    
    # Добавляем дополнительные параметры, если они указаны
    if language:
        payload["language"] = language
        
    if prompt:
        payload["prompt"] = prompt
        
    if temperature is not None:
        try:
            temp = float(temperature)
            if 0 <= temp <= 1:
                payload["temperature"] = temp
        except (ValueError, TypeError):
            pass
            
    if response_format and response_format in ["json", "text", "srt", "vtt"]:
        payload["response_format"] = response_format
        
    return payload

def prepare_tts_payload(model, input_text, voice, speed=None, format=None):
    """
    Подготавливает данные для запроса к API генерации речи из текста
    
    Args:
        model: Модель для генерации речи
        input_text: Текст для озвучивания
        voice: Голос для озвучивания
        speed: Скорость речи (опционально)
        format: Формат аудиофайла (опционально)
        
    Returns:
        dict: Данные для запроса
    """
    payload = {
        "model": "tts_1",
        "input": input_text,
        "voice": voice
    }
    
    # Добавляем дополнительные параметры, если они указаны
    if speed is not None:
        try:
            spd = float(speed)
            if 0.25 <= spd <= 4.0:
                payload["speed"] = spd
        except (ValueError, TypeError):
            pass
            
    if format and format in ["mp3", "opus", "aac", "flac"]:
        payload["response_format"] = format
        
    return payload

