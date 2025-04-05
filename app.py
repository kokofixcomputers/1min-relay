# version 1.0.1 #increment every time you make changes
# 2025-04-02 01:30 #change to actual date and time every time you make changes
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

# Импорт из наших модулей
from utils import (
    check_memcached_connection, calculate_token, api_request, set_response_headers, 
    create_session, safe_temp_file, ERROR_HANDLER, handle_options_request, split_text_for_streaming,
    safe_memcached_operation, delete_all_files_task,
    # Импорт констант
    ONE_MIN_API_URL, ONE_MIN_ASSET_URL, ONE_MIN_CONVERSATION_API_URL, 
    ONE_MIN_CONVERSATION_API_STREAMING_URL, DEFAULT_TIMEOUT, MIDJOURNEY_TIMEOUT,
    PORT, MAX_CACHE_SIZE, IMAGE_GENERATOR, IMAGE_VARIATOR,
    ALL_ONE_MIN_AVAILABLE_MODELS, VISION_SUPPORTED_MODELS, CODE_INTERPRETER_SUPPORTED_MODELS,
    RETRIEVAL_SUPPORTED_MODELS, FUNCTION_CALLING_SUPPORTED_MODELS, IMAGE_GENERATION_MODELS,
    VARIATION_SUPPORTED_MODELS, IMAGE_VARIATION_MODELS, MIDJOURNEY_ALLOWED_ASPECT_RATIOS,
    FLUX_ALLOWED_ASPECT_RATIOS, LEONARDO_ALLOWED_ASPECT_RATIOS, DALLE2_SIZES,
    DALLE3_SIZES, LEONARDO_SIZES, ALBEDO_SIZES, TEXT_TO_SPEECH_MODELS, SPEECH_TO_TEXT_MODELS,
    IMAGE_DESCRIPTION_INSTRUCTION, DOCUMENT_ANALYSIS_INSTRUCTION, 
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS, PERMIT_MODELS_FROM_SUBSET_ONLY,
    print_logo
)

from routes import text_bp, images_bp, audio_bp, files_bp

# We download the environment variables from the .env file
load_dotenv()

# Suppress warnings from flask_limiter
warnings.filterwarnings(
    "ignore", category=UserWarning, module="flask_limiter.extension"
)

# Create a logger object
logger = logging.getLogger("1min-relay")

# Install coloredlogs with desired log level
coloredlogs.install(level="DEBUG", logger=logger)

# Глобальные переменные и инициализация
app = Flask(__name__)
# Используем MEMORY_STORAGE и MEMCACHED_CLIENT из utils.memcached
from utils.memcached import MEMORY_STORAGE, MEMCACHED_CLIENT
# Очищаем хранилище при запуске
MEMORY_STORAGE.clear()
# Кэш для обработанных изображений
IMAGE_CACHE = {}  # Кэш для обработанных изображений

# Вывод логотипа
print_logo()

# Регистрируем blueprints
app.register_blueprint(text_bp)
app.register_blueprint(images_bp)
app.register_blueprint(audio_bp)
app.register_blueprint(files_bp)

# Проверка соединения с Memcached и инициализация лимитера
memcached_available, memcached_uri = check_memcached_connection()
if memcached_available:
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri=memcached_uri,
    )
    # Initialization of the client Memcache
    try:
        # First we try Pymemcache
        from pymemcache.client.base import Client

        # We extract a host and a port from URI without using. Split ('@')
        if memcached_uri.startswith('memcached://'):
            host_port = memcached_uri.replace('memcached://', '')
        else:
            host_port = memcached_uri

        # We share a host and port for Pymemcache
        if ':' in host_port:
            host, port = host_port.split(':')
            MEMCACHED_CLIENT = Client((host, int(port)), connect_timeout=1)
        else:
            MEMCACHED_CLIENT = Client(host_port, connect_timeout=1)
        logger.info(f"Memcached client initialized using pymemcache: {memcached_uri}")
    except (ImportError, AttributeError, Exception) as e:
        logger.error(f"Error initializing pymemcache client: {str(e)}")
        try:
            # If it doesn't work out, we try Python-Memcache
            # We extract a host and a port from URI without using. Split ('@')
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
    # Used for ratelimiting without memcached
    limiter = Limiter(
        get_remote_address,
        app=app,
    )
    MEMCACHED_CLIENT = None
    logger.info("Memcached not available, session storage disabled")

# Применяем лимитер к Blueprint маршрутам
# Для text_bp
limiter.limit("60 per minute")(text_bp)
# Для images_bp
limiter.limit("60 per minute")(images_bp)
# Для audio_bp
limiter.limit("60 per minute")(audio_bp)
# Для files_bp
limiter.limit("60 per minute")(files_bp)

# Получение переменных окружения для настройки моделей
one_min_models_env = os.getenv("SUBSET_OF_ONE_MIN_PERMITTED_MODELS", None)
permit_not_in_available_env = os.getenv("PERMIT_MODELS_FROM_SUBSET_ONLY", None)

# Parse or fall back to defaults
if one_min_models_env:
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS = one_min_models_env.split(",")

if permit_not_in_available_env and permit_not_in_available_env.lower() == "true":
    PERMIT_MODELS_FROM_SUBSET_ONLY = True

# Заполняем список доступных моделей из модуля constans
from utils.constants import AVAILABLE_MODELS
AVAILABLE_MODELS.clear()  # Очищаем список, если в нем что-то есть
AVAILABLE_MODELS.extend(SUBSET_OF_ONE_MIN_PERMITTED_MODELS)

# Основные настройки
# Run the task at the start of the server
if __name__ == "__main__":
    # Launch the task of deleting files
    delete_all_files_task()

    # Run the application
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
