# version 2.0.0 #increment every time you make changes
# 2025-04-04 20:30 #change to actual date and time every time you make changes
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
import datetime

import coloredlogs
import memcache
import printedcolors
import requests
import tiktoken
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response, Response, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from waitress import serve
from werkzeug.datastructures import MultiDict
from flask_cors import cross_origin

# Импорт функций и констант из utils
from utils.common import (
    calculate_token, api_request, set_response_headers, create_session,
    safe_temp_file, ERROR_HANDLER, handle_options_request, split_text_for_streaming
)
from utils.memcached import (
    check_memcached_connection, safe_memcached_operation, delete_all_files_task
)
from utils.constants import (
    # API URLs
    ONE_MIN_API_URL, ONE_MIN_ASSET_URL, ONE_MIN_CONVERSATION_API_URL, 
    ONE_MIN_CONVERSATION_API_STREAMING_URL,
    
    # Timeouts and limits
    DEFAULT_TIMEOUT, MIDJOURNEY_TIMEOUT, PORT, MAX_CACHE_SIZE,
    
    # Image related constants
    IMAGE_GENERATOR, IMAGE_VARIATOR,
    
    # Model lists and capabilities
    ALL_ONE_MIN_AVAILABLE_MODELS, VISION_SUPPORTED_MODELS, CODE_INTERPRETER_SUPPORTED_MODELS,
    RETRIEVAL_SUPPORTED_MODELS, FUNCTION_CALLING_SUPPORTED_MODELS, IMAGE_GENERATION_MODELS,
    VARIATION_SUPPORTED_MODELS, IMAGE_VARIATION_MODELS, MIDJOURNEY_ALLOWED_ASPECT_RATIOS,
    FLUX_ALLOWED_ASPECT_RATIOS, LEONARDO_ALLOWED_ASPECT_RATIOS, DALLE2_SIZES,
    DALLE3_SIZES, LEONARDO_SIZES, ALBEDO_SIZES, TEXT_TO_SPEECH_MODELS, SPEECH_TO_TEXT_MODELS,
    
    # Other constants
    IMAGE_DESCRIPTION_INSTRUCTION, DOCUMENT_ANALYSIS_INSTRUCTION, 
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS, PERMIT_MODELS_FROM_SUBSET_ONLY
)

# Импорт blueprints и функций из routes
from routes import text_bp, images_bp, audio_bp, files_bp
from routes.text import (
    format_conversation_history, get_model_capabilities, prepare_payload,
    transform_response, stream_response, emulate_stream_response
)
from routes.images import parse_aspect_ratio, retry_image_upload, create_image_variations
from routes.files import upload_document, create_conversation_with_files

# Глобальные переменные для memcached и хранилища
MEMCACHED_CLIENT = None
MEMORY_STORAGE = {}
IMAGE_CACHE = {}  # Кэш для отслеживания обработанных изображений

# Доступные модели
AVAILABLE_MODELS = []

def create_app():
    """
    Создает и настраивает Flask-приложение с использованием фабричного паттерна.
    Позволяет гибко конфигурировать приложение для разных окружений.
    
    Returns:
        Flask app: Сконфигурированное Flask-приложение
    """
    # Загружаем переменные окружения из .env файла
    load_dotenv()
    
    # Подавляем предупреждения от flask_limiter
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="flask_limiter.extension"
    )
    
    # Создаем логгер
    logger = logging.getLogger("1min-relay")
    
    # Устанавливаем coloredlogs с нужным уровнем логирования
    coloredlogs.install(level="DEBUG", logger=logger)
    
    # Создаем Flask-приложение
    app = Flask(__name__)
    
    # Инициализация Memcached и Limiter
    global MEMCACHED_CLIENT
    memcached_available, memcached_uri = check_memcached_connection()
    
    if memcached_available:
        limiter = Limiter(
            get_remote_address,
            app=app,
            storage_uri=memcached_uri,
        )
        # Инициализация клиента Memcache
        try:
            # Сначала пробуем Pymemcache
            from pymemcache.client.base import Client

            # Извлекаем хост и порт из URI
            if memcached_uri.startswith('memcached://'):
                host_port = memcached_uri.replace('memcached://', '')
            else:
                host_port = memcached_uri

            # Разбираем хост и порт для Pymemcache
            if ':' in host_port:
                host, port = host_port.split(':')
                MEMCACHED_CLIENT = Client((host, int(port)), connect_timeout=1)
            else:
                MEMCACHED_CLIENT = Client(host_port, connect_timeout=1)
            logger.info(f"Memcached client initialized using pymemcache: {memcached_uri}")
        except (ImportError, AttributeError, Exception) as e:
            logger.error(f"Error initializing pymemcache client: {str(e)}")
            try:
                # Если не получилось, пробуем Python-Memcache
                if memcached_uri.startswith('memcached://'):
                    host_port = memcached_uri.replace('memcached://', '')
                else:
                    host_port = memcached_uri

                MEMCACHED_CLIENT = memcache.Client([host_port], debug=0)
                logger.info(f"Memcached client initialized using python-memcached: {memcached_uri}")
            except (ImportError, AttributeError, Exception) as e:
                logger.error(f"Error initializing memcache client: {str(e)}")
                logger.warning(f"Failed to initialize memcached client. Session storage disabled.")
                MEMCACHED_CLIENT = None
    else:
        # Используем ratelimiting без memcached
        limiter = Limiter(
            get_remote_address,
            app=app,
        )
        MEMCACHED_CLIENT = None
        logger.info("Memcached not available, session storage disabled")
    
    # Конфигурация доступных моделей
    global AVAILABLE_MODELS
    
    # Читаем переменные окружения
    one_min_models_env = os.getenv(
        "SUBSET_OF_ONE_MIN_PERMITTED_MODELS"
    )  # e.g. "mistral-nemo,gpt-4o,deepseek-chat"
    permit_not_in_available_env = os.getenv(
        "PERMIT_MODELS_FROM_SUBSET_ONLY"
    )  # e.g. "True" or "False"

    # Разбираем или используем значения по умолчанию
    subset_models = SUBSET_OF_ONE_MIN_PERMITTED_MODELS
    if one_min_models_env:
        subset_models = one_min_models_env.split(",")

    permit_subset_only = PERMIT_MODELS_FROM_SUBSET_ONLY
    if permit_not_in_available_env and permit_not_in_available_env.lower() == "true":
        permit_subset_only = True

    # Комбинируем в единый список
    AVAILABLE_MODELS = []
    AVAILABLE_MODELS.extend(subset_models)
    
    # Регистрация blueprints
    app.register_blueprint(text_bp)
    app.register_blueprint(images_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(files_bp)
    
    return app, limiter


# Создаем приложение и инициализируем limiter
app, limiter = create_app()

# Основные настройки
# Запуск сервера при непосредственном запуске этого файла
if __name__ == "__main__":
    # Запускаем задачу удаления файлов
    delete_all_files_task()

    # Запускаем приложение
    internal_ip = socket.gethostbyname(socket.gethostname())
    try:
        response = requests.get("https://api.ipify.org")
        public_ip = response.text
    except:
        public_ip = "not found"

    logger.info(
        f"""{printedcolors.Color.fg.lightcyan}  
Server is ready to serve at:
Internal IP: {internal_ip}:{PORT}
Public IP: {public_ip} (only if you've setup port forwarding on your router.)
Enter this url to OpenAI clients supporting custom endpoint:
{internal_ip}:{PORT}/v1
If does not work, try:
{internal_ip}:{PORT}/v1/chat/completions
{printedcolors.Color.reset}"""
    )

    serve(
        app, host="0.0.0.0", port=PORT, threads=6
    )  # Thread has a default of 4 if not specified. We use 6 to increase performance and allow multiple requests at once.


