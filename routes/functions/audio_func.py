# routes/functions/audio_func.py

from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import (
    create_session, 
    api_request
)
import json

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
    
    try:
        # Проверяем структуру aiRecord (основная структура ответа)
        if "aiRecord" in response_data and "aiRecordDetail" in response_data["aiRecord"]:
            result_object = response_data["aiRecord"]["aiRecordDetail"].get("resultObject", "")
            if isinstance(result_object, list) and result_object:
                result_text = result_object[0]
            else:
                result_text = result_object
                
        # Проверяем прямую структуру resultObject
        elif "resultObject" in response_data:
            result_object = response_data["resultObject"]
            if isinstance(result_object, list) and result_object:
                result_text = result_object[0]
            else:
                result_text = result_object
        
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
    
    try:
        # Проверяем структуру aiRecord (основная структура ответа)
        if "aiRecord" in response_data and "aiRecordDetail" in response_data["aiRecord"]:
            result_object = response_data["aiRecord"]["aiRecordDetail"].get("resultObject", "")
            if isinstance(result_object, list) and result_object:
                audio_url = result_object[0]
            else:
                audio_url = result_object
                
        # Проверяем прямую структуру resultObject
        elif "resultObject" in response_data:
            result_object = response_data["resultObject"]
            if isinstance(result_object, list) and result_object:
                audio_url = result_object[0]
            else:
                audio_url = result_object
        
        if not audio_url:
            logger.error(f"[{request_id}] Could not extract audio URL from API response")
            
    except Exception as e:
        logger.error(f"[{request_id}] Error extracting audio URL from response: {str(e)}")
        
    return audio_url

