# Общие утилиты
import base64
import hashlib
import json
import logging
import os
import random
import re
import socket
import string
import tempfile
import threading
import time
import traceback
import uuid
import warnings
from io import BytesIO
from datetime import datetime

import requests
import tiktoken
from flask import jsonify, make_response, request, Response
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from requests.structures import CaseInsensitiveDict

# Создаем логгер
logger = logging.getLogger("1min-relay")

# Импорт констант
from utils.constants import (
    DEFAULT_TIMEOUT, MIDJOURNEY_TIMEOUT, 
    IMAGE_GENERATOR, IMAGE_VARIATOR
)

# Проверяем и создаем временную директорию
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temp")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Токен и константы таймаута
def calculate_token():
    """Генерирует токен на основе времени и соли.
    
    Returns:
        str: Сгенерированный токен
    """
    salt = "ABBB_salt_for_one_min_relay_123"
    token = hashlib.md5(f"{int(time.time())}_{salt}".encode()).hexdigest()
    return token

def api_request(url, method, data=None, headers=None, files=None, query_type=None, stream=False):
    """Выполняет запрос к API с правильными заголовками и обработкой ошибок.
    
    Args:
        url (str): URL для запроса
        method (str): Метод HTTP запроса ('GET', 'POST', etc.)
        data (dict, optional): Данные для отправки с запросом
        headers (dict, optional): Дополнительные заголовки запроса
        files (dict, optional): Файлы для загрузки
        query_type (str, optional): Тип запроса (например, IMAGE_GENERATOR)
        stream (bool, optional): Использовать ли потоковую передачу
        
    Returns:
        Response: Объект ответа requests
    """
    # Подготавливаем заголовки
    if headers is None:
        headers = {}
    
    payload_headers = CaseInsensitiveDict(headers)
    payload_headers["User-Agent"] = "1min-relay/1.0"
    
    # Добавляем токен для авторизации
    payload_headers["x-api-token"] = calculate_token()
    
    # Определяем таймаут на основе типа запроса
    timeout = DEFAULT_TIMEOUT
    if query_type == IMAGE_GENERATOR or query_type == IMAGE_VARIATOR:
        if data and isinstance(data, dict) and data.get("model") == "midjourney":
            timeout = MIDJOURNEY_TIMEOUT
    
    try:
        # Выполняем запрос в зависимости от метода
        if method.upper() == "GET":
            response = requests.get(
                url,
                headers=payload_headers,
                params=data,
                timeout=timeout,
                stream=stream,
            )
        elif method.upper() == "POST":
            if files:
                response = requests.post(
                    url,
                    headers=payload_headers,
                    data=data,
                    files=files,
                    timeout=timeout,
                    stream=stream,
                )
            else:
                payload_headers["Content-Type"] = "application/json"
                response = requests.post(
                    url,
                    headers=payload_headers,
                    json=data,
                    timeout=timeout,
                    stream=stream,
                )
        elif method.upper() == "DELETE":
            response = requests.delete(
                url,
                headers=payload_headers,
                json=data,
                timeout=timeout,
                stream=stream,
            )
        else:
            return {
                "error": {
                    "message": f"Unsupported method: {method}",
                    "type": "relay_error",
                }
            }
        
        # Возвращаем объект ответа
        return response
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in API request: {str(e)}")
        print(error_traceback)
        return {
            "error": {
                "message": f"API request error: {str(e)}",
                "type": "relay_error",
                "traceback": error_traceback,
            }
        }

def set_response_headers(response):
    """Устанавливает заголовки CORS для ответа HTTP.
    
    Args:
        response: Объект ответа Flask
        
    Returns:
        response: Объект ответа с добавленными заголовками
    """
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add(
        "Access-Control-Allow-Headers",
        "Origin, X-Requested-With, Content-Type, Accept, Authorization",
    )
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
    return response

def create_session():
    """Создает новую сессию с уникальным ID.
    
    Returns:
        str: ID сессии
    """
    return str(uuid.uuid4())

def safe_temp_file(content=None, extension=".tmp"):
    """Создает временный файл с заданным содержимым и расширением.
    
    Args:
        content: Содержимое файла
        extension (str): Расширение файла
        
    Returns:
        str: Путь к созданному файлу
    """
    # Создаем уникальное имя файла на основе timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = uuid.uuid4().hex[:8]
    filename = f"temp_{timestamp}_{random_suffix}{extension}"
    filepath = os.path.join(TEMP_DIR, filename)
    
    # Записываем содержимое в файл
    if content is not None:
        if isinstance(content, bytes):
            with open(filepath, "wb") as f:
                f.write(content)
        elif isinstance(content, str):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            with open(filepath, "wb") as f:
                f.write(content.read())
    
    return filepath

# Обработчик ошибок для всех конечных точек
def ERROR_HANDLER(error_str, status_code=500):
    """Создает стандартизированный ответ об ошибке.
    
    Args:
        error_str (str): Сообщение об ошибке
        status_code (int): HTTP код статуса
        
    Returns:
        Response: Объект ответа Flask с информацией об ошибке
    """
    response = make_response(
        jsonify(
            {
                "error": {
                    "message": error_str,
                    "type": "relay_error",
                    "code": status_code,
                }
            }
        ),
        status_code,
    )
    return set_response_headers(response)

def handle_options_request():
    """Обрабатывает OPTIONS запросы для CORS preflight.
    
    Returns:
        Response: Объект ответа Flask с правильными CORS заголовками
    """
    response = make_response()
    return set_response_headers(response)

def split_text_for_streaming(text):
    """Разделяет текст на части для потоковой передачи.
    
    Args:
        text (str): Текст для разделения
        
    Returns:
        list: Список частей текста
    """
    result = []
    
    # Пытаемся разделить текст по параграфам, предложениям или словам
    paragraphs = re.split(r"\n\s*\n", text)
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Если предложение слишком длинное, разбиваем на части
            if len(sentence) > 150:
                parts = []
                words = sentence.split()
                current_part = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > 150:
                        parts.append(" ".join(current_part))
                        current_part = [word]
                        current_length = len(word)
                    else:
                        current_part.append(word)
                        current_length += len(word) + 1
                
                if current_part:
                    parts.append(" ".join(current_part))
                
                result.extend(parts)
            else:
                result.append(sentence)
    
    return result
