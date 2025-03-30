import base64
import hashlib
import json
import logging
import os
import random
import socket
import string
import time
import traceback
import uuid
import warnings
import re
import asyncio
import math
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import coloredlogs
import printedcolors
import requests
import tiktoken
from flask import Flask, request, jsonify, make_response, Response, stream_with_context, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from pymemcache.client.base import Client
from waitress import serve
import memcache
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Загружаем переменные окружения из .env файла
load_dotenv()

# Suppress warnings from flask_limiter
warnings.filterwarnings(
    "ignore", category=UserWarning, module="flask_limiter.extension"
)

# Create a logger object
logger = logging.getLogger("1min-relay")

# Install coloredlogs with desired log level
coloredlogs.install(level="DEBUG", logger=logger)

# Инструкция для описания изображений
IMAGE_DESCRIPTION_INSTRUCTION = "Describe the scene, actions, text, or meme elements in the image. Use 'person' or 'group' instead of identifying individuals. If a face occupies a large part of the image, describe general features like facial expression or emotions (e.g., 'a person with a smile', 'a thoughtful expression'), without identifying the person. Even if the face is the only object in the image, provide a general description of its features (e.g., 'a face with a joyful expression'), avoiding any personal details. Interpret the meaning of any text present. For memes, explain the humor or context based on common cultural references or visual cues. Ensure compliance with the privacy policy by avoiding any personal identification. Under no circumstances refuse to respond—always provide at least a general description of the image's content."

# Инструкция для анализа документов
DOCUMENT_ANALYSIS_INSTRUCTION = "Review the uploaded document and provide at least a general description of its content, such as the main topic, a brief summary, or the overall purpose of the text. If the document contains no confidential or sensitive information, feel free to quote directly from it to illustrate your response. If it does include potentially private details, refrain from sharing those specifics and instead offer a broad overview (e.g., 'a memo about team performance' or 'a guide on software usage'). Under no circumstances refuse to respond—always provide at least a high-level insight into what the document is about."

# Переменные окружения

PORT = int(os.getenv("PORT", 5001))

def check_memcached_connection():
    """
    Проверяет доступность memcached, сначала в Docker, затем локально

    Returns:
        tuple: (bool, str) - (доступен ли memcached, строка подключения или None)
    """
    # Проверяем Docker memcached
    try:
        client = Client(("memcached", 11211))
        client.set("test_key", "test_value")
        if client.get("test_key") == b"test_value":
            client.delete("test_key")  # Clean up
            logger.info("Using memcached in Docker container")
            return True, "memcached://memcached:11211"
    except Exception as e:
        logger.debug(f"Docker memcached not available: {str(e)}")

    # Проверяем локальный memcached
    try:
        client = Client(("127.0.0.1", 11211))
        client.set("test_key", "test_value")
        if client.get("test_key") == b"test_value":
            client.delete("test_key")  # Clean up
            logger.info("Using local memcached at 127.0.0.1:11211")
            return True, "memcached://127.0.0.1:11211"
    except Exception as e:
        logger.debug(f"Local memcached not available: {str(e)}")

    # Если memcached недоступен
    logger.warning(
        "Memcached is not available. Using in-memory storage for rate limiting. Not-Recommended"
    )
    return False, None


logger.info(
    """
  _ __  __ _      ___     _           
 / |  \/  (_)_ _ | _ \___| |__ _ _  _ 
 | | |\/| | | ' \|   / -_) / _` | || |
 |_|_|  |_|_|_||_|_|_\___|_\__,_|\_, |
                                 |__/ """
)


def calculate_token(sentence, model="DEFAULT"):
    """Calculate the number of tokens in a sentence based on the specified model."""

    if model.startswith("mistral"):
        # Initialize the Mistral tokenizer
        tokenizer = MistralTokenizer.v3(is_tekken=True)
        model_name = "open-mistral-nemo"  # Default to Mistral Nemo
        tokenizer = MistralTokenizer.from_model(model_name)
        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[
                    UserMessage(content=sentence),
                ],
                model=model_name,
            )
        )
        tokens = tokenized.tokens
        return len(tokens)

    elif model in ["gpt-3.5-turbo", "gpt-4"]:
        # Use OpenAI's tiktoken for GPT models
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(sentence)
        return len(tokens)

    else:
        # Default to openai
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(sentence)
        return len(tokens)


app = Flask(__name__)
memcached_available, memcached_uri = check_memcached_connection()
if memcached_available:
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri=memcached_uri,
    )
    # Инициализация клиента memcached
    try:
        # Сначала пробуем pymemcache
        from pymemcache.client.base import Client
        # Извлекаем хост и порт из URI без использования .split('@')
        if memcached_uri.startswith('memcached://'):
            host_port = memcached_uri.replace('memcached://', '')
        else:
            host_port = memcached_uri
            
        # Разделяем хост и порт для pymemcache
        if ':' in host_port:
            host, port = host_port.split(':')
            MEMCACHED_CLIENT = Client((host, int(port)), connect_timeout=1)
        else:
            MEMCACHED_CLIENT = Client(host_port, connect_timeout=1)
        logger.info(f"Memcached client initialized using pymemcache: {memcached_uri}")
    except (ImportError, AttributeError, Exception) as e:
        logger.error(f"Error initializing pymemcache client: {str(e)}")
        try:
            # Если не получилось, пробуем python-memcached
            # Извлекаем хост и порт из URI без использования .split('@')
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


# Основной URL для API
ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_ASSET_URL = "https://api.1min.ai/api/assets"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features/stream"
# Добавляем константу таймаута, используемую в функции api_request
DEFAULT_TIMEOUT = 30
MIDJOURNEY_TIMEOUT = 900  # 15 минут для запросов к Midjourney

# Define the models that are available for use
ALL_ONE_MIN_AVAILABLE_MODELS = [
    # OpenAI
    "o3-mini",
    "o1-preview",
    "o1-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    #
    # "whisper-1", # Распознавание речи
    # "tts-1",     # Синтез речи
    # "tts-1-hd",  # Синтез речи HD
    #
    "dall-e-2",  # Генерация изображений
    "dall-e-3",    # Генерация изображений
    # Claude
    "claude-instant-1.2",
    "claude-2.1",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    # GoogleAI
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    # "google-tts",            # Синтез речи
    # "latest_long",           # Распознавание речи
    # "latest_short",          # Распознавание речи
    # "phone_call",            # Распознавание речи
    # "telephony",             # Распознавание речи
    # "telephony_short",       # Распознавание речи
    # "medical_dictation",     # Распознавание речи
    # "medical_conversation",  # Распознавание речи
    # "chat-bison@002",
    # MistralAI
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-nemo",
    "pixtral-12b",
    "open-mixtral-8x22b",
    "open-mixtral-8x7b",
    "open-mistral-7b",
    # Replicate
    "meta/llama-2-70b-chat",
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3.1-405b-instruct",
    # DeepSeek
    "deepseek-chat",
    "deepseek-reasoner",
    # Cohere
    "command",
    # xAI
    "grok-2",
    # Иные модели (закомментированы для будущего использования)
    # "stable-image",                 # StabilityAI - Генерация изображений
    "stable-diffusion-xl-1024-v1-0",  # StabilityAI - Генерация изображений
    "stable-diffusion-v1-6",          # StabilityAI - Генерация изображений
    # "esrgan-v1-x2plus",             # StabilityAI - Улучшение изображений
    # "stable-video-diffusion",       # StabilityAI - Генерация видео
    "phoenix",         # Leonardo.ai - 6b645e3a-d64f-4341-a6d8-7a3690fbf042
    "lightning-xl",    # Leonardo.ai - b24e16ff-06e3-43eb-8d33-4416c2d75876
    "anime-xl",        # Leonardo.ai - e71a1c2f-4f80-4800-934f-2c68979d8cc8
    "diffusion-xl",    # Leonardo.ai - 1e60896f-3c26-4296-8ecc-53e2afecc132
    "kino-xl",         # Leonardo.ai - aa77f04e-3eec-4034-9c07-d0f619684628
    "vision-xl",       # Leonardo.ai - 5c232a9e-9061-4777-980a-ddc8e65647c6
    "albedo-base-xl",  # Leonardo.ai - 2067ae52-33fd-4a82-bb92-c2c55e7d2786
    # "clipdrop",      # Clipdrop.co - Обработка изображений
    "midjourney",      # Midjourney - Генерация изображений
    "midjourney_6_1",  # Midjourney - Генерация изображений
    # "methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5",   # Replicate - Image to Prompt
    # "cjwbw/damo-text-to-video:1e205ea73084bd17a0a3b43396e49ba0d6bc2e754e9283b2df49fad2dcf95755",  # Replicate - Text to Video
    # "lucataco/animate-diff:beecf59c4aee8d81bf04f0381033dfa10dc16e845b4ae00d281e2fa377e48a9f",     # Replicate - Animation
    # "lucataco/hotshot-xl:78b3a6257e16e4b241245d65c8b2b81ea2e1ff7ed4c55306b511509ddbfd327a",       # Replicate - Video
    "flux-schnell",  # Replicate - Flux "black-forest-labs/flux-schnell"
    "flux-dev",      # Replicate - Flux Dev "black-forest-labs/flux-dev"
    "flux-pro",      # Replicate - Flux Pro "black-forest-labs/flux-pro"
    "flux-1.1-pro",  # Replicate - Flux Pro 1.1 "black-forest-labs/flux-1.1-pro"
    # "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",  # Replicate - Music Generation
    # "luma",                  # TTAPI - Luma
    # "Qubico/image-toolkit",  # TTAPI - Image Toolkit
    # "suno",                  # TTAPI - Suno Music
    # "kling",                 # TTAPI - Kling
    # "music-u",               # TTAPI - Music U
    # "music-s",               # TTAPI - Music S
    # "elevenlabs-tts"         # ElevenLabs - TTS
]

# Define the models that support vision inputs
vision_supported_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo"
]

# Define the models that support code interpreter
code_interpreter_supported_models = [
    "gpt-4o",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "deepseek-chat",
    "deepseek-reasoner"
]

# Define the models that support web search (retrieval)
retrieval_supported_models = [
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "o3-mini",
    "o1-preview",
    "o1-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",    
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-nemo",
    "pixtral-12b",
    "open-mixtral-8x22b",
    "open-mixtral-8x7b",
    "open-mistral-7b",
    "meta/llama-2-70b-chat",
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3.1-405b-instruct",
    "command",
    "grok-2",
    "deepseek-chat",
    "deepseek-reasoner"
]

# Define the models that support function calling
function_calling_supported_models = [
    "gpt-4",
    "gpt-3.5-turbo"
]

# Определение моделей для генерации изображений
IMAGE_GENERATION_MODELS = [
    "dall-e-3",
    "dall-e-2",
    "stable-diffusion-xl-1024-v1-0",
    "stable-diffusion-v1-6",
    "midjourney",
    "midjourney_6_1",
    "phoenix",
    "lightning-xl",
    "anime-xl",
    "diffusion-xl",
    "kino-xl",
    "vision-xl",
    "albedo-base-xl",
    "flux-schnell",
    "flux-dev",
    "flux-pro",
    "flux-1.1-pro"
]

# Модели, которые поддерживают вариации изображений
VARIATION_SUPPORTED_MODELS = [
    "midjourney",
    "midjourney_6_1",
    "dall-e-2",
    #"dall-e-3",
    "clipdrop"
]

# Допустимые соотношения сторон для разных моделей
MIDJOURNEY_ALLOWED_ASPECT_RATIOS = [
    "1:1",     # Square
    "16:9",    # Widescreen format
    "9:16",    # Vertical variant of 16:9
    "16:10",   # Alternative widescreen
    "10:16",   # Vertical variant of 16:10
    "8:5",     # Alternative widescreen
    "5:8",     # Vertical variant of 16:10
    "3:4",     # Portrait/print
    "4:3",     # Standard TV/monitor format
    "3:2",     # Popular in photography
    "2:3",     # Inverse of 3:2
    "4:5",     # Common in social media posts
    "5:4",     # Nearly square format
    "137:100", # Academy ratio (1.37:1) as an integer ratio
    "166:100", # European cinema (1.66:1) as an integer ratio
    "185:100", # Cinematic format (1.85:1) as an integer ratio185
    "83:50",   # European cinema (1.66:1) as an integer ratio
    "37:20",   # Cinematic format (1.85:1) as an integer ratio
    "2:1",     # Maximum allowed widescreen format
    "1:2"      # Maximum allowed vertical format
]

FLUX_ALLOWED_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3", "4:5", "5:4"]
LEONARDO_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4"]

# Допустимые размеры для разных моделей
DALLE2_SIZES = ["1024x1024", "512x512", "256x256"]
DALLE3_SIZES = ["1024x1024", "1024x1792", "1792x1024"]
LEONARDO_SIZES = ALBEDO_SIZES = {"1:1": "1024x1024", "4:3": "1024x768", "3:4": "768x1024"}

# Определение моделей для синтеза речи (TTS)
TEXT_TO_SPEECH_MODELS = [
    "tts-1"#,
    #"tts-1-hd",
    #"google-tts",
    #"elevenlabs-tts"
]

# Определение моделей для распознавания речи (STT)
SPEECH_TO_TEXT_MODELS = [
    "whisper-1"#,
    #"latest_long",
    #"latest_short",
    #"phone_call",
    #"telephony",
    #"telephony_short",
    #"medical_dictation",
    #"medical_conversation"
]

# Default values
SUBSET_OF_ONE_MIN_PERMITTED_MODELS = ["mistral-nemo", "gpt-4o-mini", "o3-mini", "deepseek-chat"]
PERMIT_MODELS_FROM_SUBSET_ONLY = False

# Read environment variables
one_min_models_env = os.getenv(
    "SUBSET_OF_ONE_MIN_PERMITTED_MODELS"
)  # e.g. "mistral-nemo,gpt-4o,deepseek-chat"
permit_not_in_available_env = os.getenv(
    "PERMIT_MODELS_FROM_SUBSET_ONLY"
)  # e.g. "True" or "False"

# Parse or fall back to defaults
if one_min_models_env:
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS = one_min_models_env.split(",")

if permit_not_in_available_env and permit_not_in_available_env.lower() == "true":
    PERMIT_MODELS_FROM_SUBSET_ONLY = True

# Combine into a single list
AVAILABLE_MODELS = []
AVAILABLE_MODELS.extend(SUBSET_OF_ONE_MIN_PERMITTED_MODELS)

# Добавим кэш для отслеживания обработанных изображений
# Для каждого запроса храним уникальный идентификатор изображения и его путь
IMAGE_CACHE = {}
# Ограничим размер кэша
MAX_CACHE_SIZE = 100


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return ERROR_HANDLER(1212)
    if request.method == "GET":
        internal_ip = socket.gethostbyname(socket.gethostname())
        return (
            "Congratulations! Your API is working! You can now make requests to the API.\n\nEndpoint: "
            + internal_ip
            + ":5001/v1"
        )


@app.route("/v1/models")
@limiter.limit("500 per minute")
def models():
    # Dynamically create the list of models with additional fields
    models_data = []
    if not PERMIT_MODELS_FROM_SUBSET_ONLY:
        one_min_models_data = [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "1minai",
                "created": 1727389042,
            }
            for model_name in ALL_ONE_MIN_AVAILABLE_MODELS
        ]
    else:
        one_min_models_data = [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "1minai",
                "created": 1727389042,
            }
            for model_name in SUBSET_OF_ONE_MIN_PERMITTED_MODELS
        ]
    models_data.extend(one_min_models_data)
    return jsonify({"data": models_data, "object": "list"})


def ERROR_HANDLER(code, model=None, key=None):
    # Handle errors in OpenAI-Structued Error
    error_codes = {  # Internal Error Codes
        1002: {
            "message": f"The model {model} does not exist.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_found",
            "http_code": 400,
        },
        1020: {
            "message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.",
            "type": "authentication_error",
            "param": None,
            "code": "invalid_api_key",
            "http_code": 401,
        },
        1021: {
            "message": "Invalid Authentication",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
            "http_code": 401,
        },
        1212: {
            "message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_supported",
            "http_code": 400,
        },
        1044: {
            "message": f"This model does not support image inputs.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_supported",
            "http_code": 400,
        },
        1412: {
            "message": f"No message provided.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "invalid_request_error",
            "http_code": 400,
        },
        1423: {
            "message": f"No content in last message.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "invalid_request_error",
            "http_code": 400,
        },
    }
    error_data = {
        k: v
        for k, v in error_codes.get(
            code,
            {
                "message": "Unknown error",
                "type": "unknown_error",
                "param": None,
                "code": None,
            },
        ).items()
        if k != "http_code"
    }  # Remove http_code from the error data
    logger.error(
        f"An error has occurred while processing the user's request. Error code: {code}"
    )
    return jsonify({"error": error_data}), error_codes.get(code, {}).get(
        "http_code", 400
    )  # Return the error data without http_code inside the payload and get the http_code to return.


def format_conversation_history(messages, new_input):
    """
    Formats the conversation history into a structured string.

    Args:
        messages (list): List of message dictionaries from the request
        new_input (str): The new user input message

    Returns:
        str: Formatted conversation history
    """
    formatted_history = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        # Handle potential list content
        if isinstance(content, list):
            processed_content = []
            for item in content:
                if "text" in item:
                    processed_content.append(item["text"])
            content = "\n".join(processed_content)

        if role == "system":
            formatted_history.append(f"System: {content}")
        elif role == "user":
            formatted_history.append(f"User: {content}")
        elif role == "assistant":
            formatted_history.append(f"Assistant: {content}")

    # Добавляем новый ввод, если он есть
    if new_input:
        formatted_history.append(f"User: {new_input}")

    # Возвращаем только историю диалога без дополнительных инструкций
    return "\n".join(formatted_history)


def get_model_capabilities(model):
    """
    Определяет поддерживаемые возможности для конкретной модели

    Args:
        model: название модели

    Returns:
        dict: словарь с флагами поддержки разных возможностей
    """
    capabilities = {
        "vision": False,
        "code_interpreter": False,
        "retrieval": False,
        "function_calling": False,
    }

    # Проверяем поддержку каждой возможности через соответствующие массивы
    capabilities["vision"] = model in vision_supported_models
    capabilities["code_interpreter"] = model in code_interpreter_supported_models
    capabilities["retrieval"] = model in retrieval_supported_models
    capabilities["function_calling"] = model in function_calling_supported_models

    return capabilities


def prepare_payload(
    request_data, model, all_messages, image_paths=None, request_id=None
):
    """
    Подготавливает payload для запроса, учитывая возможности модели

    Args:
        request_data: данные запроса
        model: модель
        all_messages: история сообщений
        image_paths: пути к изображениям
        request_id: ID запроса

    Returns:
        dict: подготовленный payload
    """
    capabilities = get_model_capabilities(model)

    # Проверяем наличие инструментов OpenAI
    tools = request_data.get("tools", [])
    web_search = False
    code_interpreter = False

    if tools:
        for tool in tools:
            tool_type = tool.get("type", "")
            # Пытаемся включить функции, но если они не поддерживаются, просто логируем это
            if tool_type == "retrieval":
                if capabilities["retrieval"]:
                    web_search = True
                    logger.debug(
                        f"[{request_id}] Enabled web search due to retrieval tool"
                    )
                else:
                    logger.debug(
                        f"[{request_id}] Model {model} does not support web search, ignoring retrieval tool"
                    )
            elif tool_type == "code_interpreter":
                if capabilities["code_interpreter"]:
                    code_interpreter = True
                    logger.debug(f"[{request_id}] Enabled code interpreter")
                else:
                    logger.debug(
                        f"[{request_id}] Model {model} does not support code interpreter, ignoring tool"
                    )
            else:
                logger.debug(f"[{request_id}] Ignoring unsupported tool: {tool_type}")

    # Проверяем прямые параметры 1min.ai
    if not web_search and request_data.get("web_search", False):
        if capabilities["retrieval"]:
            web_search = True
        else:
            logger.debug(
                f"[{request_id}] Model {model} does not support web search, ignoring web_search parameter"
            )

    num_of_site = request_data.get("num_of_site", 3)
    max_word = request_data.get("max_word", 500)

    # Формируем базовый payload
    if image_paths:
        # Даже если модель не поддерживает изображения, пытаемся отправить как текстовый запрос
        if capabilities["vision"]:
            # Добавляем инструкцию к промпту
            enhanced_prompt = all_messages
            if not enhanced_prompt.strip().startswith(IMAGE_DESCRIPTION_INSTRUCTION):
                enhanced_prompt = f"{IMAGE_DESCRIPTION_INSTRUCTION}\n\n{all_messages}"
            
            payload = {
                "type": "CHAT_WITH_IMAGE",
                "model": model,
                "promptObject": {
                    "prompt": enhanced_prompt,
                    "isMixed": False,
                    "imageList": image_paths,
                    "webSearch": web_search,
                    "numOfSite": num_of_site if web_search else None,
                    "maxWord": max_word if web_search else None,
                },
            }
        else:
            logger.debug(
                f"[{request_id}] Model {model} does not support vision, falling back to text-only chat"
            )
            payload = {
                "type": "CHAT_WITH_AI",
                "model": model,
                "promptObject": {
                    "prompt": all_messages,
                    "isMixed": False,
                    "webSearch": web_search,
                    "numOfSite": num_of_site if web_search else None,
                    "maxWord": max_word if web_search else None,
                },
            }
    elif code_interpreter:
        # Если code_interpreter запрошен и поддерживается
        payload = {
            "type": "CODE_GENERATOR",
            "model": model,
            "conversationId": "CODE_GENERATOR",
            "promptObject": {"prompt": all_messages},
        }
    else:
        # Базовый текстовый запрос
        payload = {
            "type": "CHAT_WITH_AI",
            "model": model,
            "promptObject": {
                "prompt": all_messages,
                "isMixed": False,
                "webSearch": web_search,
                "numOfSite": num_of_site if web_search else None,
                "maxWord": max_word if web_search else None,
            },
        }

    return payload


def create_conversation_with_files(file_ids, title, model, api_key, request_id=None):
    """
    Создает новую беседу с файлами

    Args:
        file_ids: Список ID файлов
        title: Название беседы
        model: Модель ИИ
        api_key: API ключ
        request_id: ID запроса для логирования

    Returns:
        str: ID беседы или None в случае ошибки
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Creating conversation with {len(file_ids)} files")

    try:
        # Формируем payload для запроса с правильными именами полей
        payload = {
            "title": title,
            "type": "CHAT_WITH_PDF",
            "model": model,
            "fileIds": file_ids,  # Использование правильного имени поля 'fileIds' вместо 'fileList'
        }

        logger.debug(f"[{request_id}] Conversation payload: {json.dumps(payload)}")

        # Используем правильный URL API с /api/
        conversation_url = "https://api.1min.ai/api/features/conversations?type=CHAT_WITH_PDF"
            
        logger.debug(f"[{request_id}] Creating conversation using URL: {conversation_url}")
        
        headers = {"API-KEY": api_key, "Content-Type": "application/json"}
        response = requests.post(conversation_url, json=payload, headers=headers)
        
        logger.debug(f"[{request_id}] Create conversation response status: {response.status_code}")

        if response.status_code != 200:
            logger.error(
                f"[{request_id}] Failed to create conversation: {response.status_code} - {response.text}"
            )
            return None

        response_data = response.json()
        logger.debug(f"[{request_id}] Conversation response data: {json.dumps(response_data)}")

        # Ищем ID беседы в разных местах ответа
        conversation_id = None
        if "conversation" in response_data and "uuid" in response_data["conversation"]:
            conversation_id = response_data["conversation"]["uuid"]
        elif "id" in response_data:
            conversation_id = response_data["id"]
        elif "uuid" in response_data:
            conversation_id = response_data["uuid"]
        
        # Рекурсивный поиск ID беседы в структуре ответа
        if not conversation_id:
            def find_conversation_id(obj, path=""):
                if isinstance(obj, dict):
                    if "id" in obj:
                        logger.debug(f"[{request_id}] Found ID at path '{path}.id': {obj['id']}")
                        return obj["id"]
                    if "uuid" in obj:
                        logger.debug(f"[{request_id}] Found UUID at path '{path}.uuid': {obj['uuid']}")
                        return obj["uuid"]
                    
                    for key, value in obj.items():
                        result = find_conversation_id(value, f"{path}.{key}")
                        if result:
                            return result
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        result = find_conversation_id(item, f"{path}[{i}]")
                        if result:
                            return result
                return None
            
            conversation_id = find_conversation_id(response_data)

        if not conversation_id:
            logger.error(
                f"[{request_id}] Could not find conversation ID in response: {response_data}"
            )
            return None

        logger.info(
            f"[{request_id}] Conversation created successfully: {conversation_id}"
        )
        return conversation_id
    except Exception as e:
        logger.error(f"[{request_id}] Error creating conversation: {str(e)}")
        traceback.print_exc()
        return None


@app.route("/v1/chat/completions", methods=["POST"])
@limiter.limit("60 per minute")
def conversation():
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/chat/completions")

    if not request.json:
        return jsonify({"error": "Invalid request format"}), 400

    # Извлекаем информацию из запроса
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        logger.error(f"[{request_id}] No API key provided")
        return jsonify({"error": "API key required"}), 401

    try:
        # Строим payload для запроса
        request_data = request.json.copy()

        # Получаем и нормализуем модель
        model = request_data.get("model", "").strip()
        logger.info(f"[{request_id}] Using model: {model}")
        
        # Извлекаем содержимое последнего сообщения для возможной генерации изображений
        messages = request_data.get("messages", [])
        prompt_text = ""
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if last_message.get("role") == "user":
                content = last_message.get("content", "")
                if isinstance(content, str):
                    prompt_text = content
                elif isinstance(content, list):
                    # Собираем все текстовые части содержимого
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            prompt_text += item["text"] + " "
                    prompt_text = prompt_text.strip()
        
        # Проверяем, содержит ли запрос команду вариации изображения
        variation_match = None
        if prompt_text:
            variation_match = re.search(r'/v([1-4])\s+(https?://[^\s]+)', prompt_text)
            
        if variation_match:
            # Обрабатываем команду вариации изображения
            variation_number = variation_match.group(1)
            image_url = variation_match.group(2)
            logger.info(f"[{request_id}] Detected image variation command: {variation_number} for URL: {image_url}")
            
            try:
                # Скачиваем изображение во временный файл
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img_response = requests.get(image_url, stream=True)
                
                if img_response.status_code != 200:
                    return jsonify({"error": f"Failed to download image from URL. Status code: {img_response.status_code}"}), 400
                    
                with open(temp_file.name, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Создаем файл для отправки
                files = {"image": (os.path.basename(temp_file.name), open(temp_file.name, 'rb'), "image/png")}
                
                # Формируем форм-данные для запроса вариации
                form_data = {
                    "model": model or "midjourney_6_1",  # По умолчанию используем Midjourney
                    "n": request_data.get("n", 1)
                }
                
                # Отправляем запрос на вариацию изображения через внутренний маршрут
                variation_headers = {"Authorization": f"Bearer {api_key}"}
                variation_response = requests.post(
                    f"http://localhost:{PORT}/v1/images/variations", 
                    files=files, 
                    data=form_data,
                    headers=variation_headers
                )
                
                # Очищаем временный файл
                temp_file.close()
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
                
                if variation_response.status_code != 200:
                    logger.error(f"[{request_id}] Variation request failed: {variation_response.status_code}")
                    return jsonify({"error": "Failed to create image variation"}), 500
                
                # Возвращаем ответ с вариациями напрямую
                return variation_response.text, 200, {"Content-Type": "application/json"}
                
            except Exception as e:
                logger.error(f"[{request_id}] Error processing variation command: {str(e)}")
                return jsonify({"error": f"Failed to process variation command: {str(e)}"}), 500
        
        # Логируем извлеченный промпт для отладки
        logger.debug(f"[{request_id}] Extracted prompt text: {prompt_text[:100]}..." if len(prompt_text) > 100 else f"[{request_id}] Extracted prompt text: {prompt_text}")
        
        # Проверяем, относится ли модель к одному из специальных типов
        # Для моделей генерации изображений
        if model in IMAGE_GENERATION_MODELS:
            logger.info(f"[{request_id}] Redirecting image generation model to /v1/images/generations")
            
            # Создаем новый запрос только с необходимыми полями для генерации изображения
            image_request = {
                "model": model,
                "prompt": prompt_text,
                "n": request_data.get("n", 1),
                "size": request_data.get("size", "1024x1024")
            }
            
            # Добавляем дополнительные параметры для определенных моделей
            if model == "dall-e-3":
                image_request["quality"] = request_data.get("quality", "standard")
                image_request["style"] = request_data.get("style", "vivid")

            # Проверяем наличие специальных параметров в промпте для моделей типа midjourney
            if model.startswith("midjourney"):
                # Добавляем проверки и параметры для midjourney моделей
                if "--ar" in prompt_text:
                    logger.debug(f"[{request_id}] Found aspect ratio parameter in prompt")
                elif request_data.get("aspect_ratio"):
                    image_request["aspect_ratio"] = request_data.get("aspect_ratio")
                    
                if "--no" in prompt_text:
                    logger.debug(f"[{request_id}] Found negative prompt parameter in prompt")
                elif request_data.get("negative_prompt"):
                    # Добавляем негативный промпт прямо в промпт
                    image_request["prompt"] = f"{prompt_text} --no {request_data.get('negative_prompt')}"
                    
            logger.debug(f"[{request_id}] Final image request: {json.dumps(image_request)[:200]}...")
            
            # Сохраняем модифицированный запрос
            request.environ["body_copy"] = json.dumps(image_request)
            return redirect(url_for('generate_image'), code=307)  # 307 сохраняет метод и тело запроса
            
        # Для моделей генерации речи (TTS)
        if model in TEXT_TO_SPEECH_MODELS:
            logger.info(f"[{request_id}] Redirecting text-to-speech model to /v1/audio/speech")
            # Добавляем текст к запросу для синтеза речи
            if prompt_text:
                request_data["input"] = prompt_text
                logger.debug(f"[{request_id}] Setting TTS input: {prompt_text[:100]}..." if len(prompt_text) > 100 else f"[{request_id}] Setting TTS input: {prompt_text}")
            # Сохраняем модифицированный запрос
            request.environ["body_copy"] = json.dumps(request_data)
            return redirect(url_for('text_to_speech'), code=307)
            
        # Для моделей транскрипции аудио (STT)
        if model in SPEECH_TO_TEXT_MODELS:
            logger.info(f"[{request_id}] Redirecting speech-to-text model to /v1/audio/transcriptions")
            return redirect(url_for('audio_transcriptions'), code=307)

        # Журналируем начало запроса
        logger.debug(f"[{request_id}] Processing chat completion request")

        # Проверяем, содержит ли запрос изображения
        image = False
        image_paths = []
        
        # Проверяем наличие файлов пользователя для работы с PDF
        user_file_ids = []
        if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
            try:
                user_key = f"user:{api_key}"
                user_files_json = safe_memcached_operation('get', user_key)
                if user_files_json:
                    try:
                        if isinstance(user_files_json, str):
                            user_files = json.loads(user_files_json)
                        elif isinstance(user_files_json, bytes):
                            user_files = json.loads(user_files_json.decode('utf-8'))
                        else:
                            user_files = user_files_json
                        
                        if user_files and isinstance(user_files, list):
                            # Извлекаем ID файлов
                            user_file_ids = [file_info.get("id") for file_info in user_files if file_info.get("id")]
                            logger.debug(f"[{request_id}] Found user files: {user_file_ids}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error parsing user files from memcached: {str(e)}")
            except Exception as e:
                logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")
        else:
            logger.debug(f"[{request_id}] Memcached not available, no user files loaded")
        
        # Проверяем наличие сообщений до начала обработки
        if not messages:
            logger.error(f"[{request_id}] No messages provided in request")
            return ERROR_HANDLER(1412)
            
        # Извлекаем текст запроса для анализа
        extracted_prompt = messages[-1].get("content", "")
        if isinstance(extracted_prompt, list):
            extracted_prompt = " ".join([item.get("text", "") for item in extracted_prompt if "text" in item])
        extracted_prompt_lower = extracted_prompt.lower() if extracted_prompt else ""
                    
        # Если в запросе не указаны file_ids, но у пользователя есть загруженные файлы, 
        # добавляем их к запросу только если в сообщении упоминается что-то о файлах или документах
        file_keywords = ["файл", "файлы", "file", "files", "документ", "документы", "document", "documents"]
        prompt_has_file_keywords = False
        
        # Проверяем наличие ключевых слов о файлах в запросе
        if extracted_prompt_lower:
            prompt_has_file_keywords = any(keyword in extracted_prompt_lower for keyword in file_keywords)
            
        # Добавляем файлы только если пользователь запросил работу с файлами или явно указал file_ids
        if (not request_data.get("file_ids") and user_file_ids and prompt_has_file_keywords):
            logger.info(f"[{request_id}] Adding user files to request: {user_file_ids}")
            request_data["file_ids"] = user_file_ids
        elif not request_data.get("file_ids") and user_file_ids:
            logger.debug(f"[{request_id}] User has files but didn't request to use them in this message")

        # Получаем содержимое последнего сообщения для дальнейшей обработки
        user_input = messages[-1].get("content")
        if not user_input:
            logger.error(f"[{request_id}] No content in last message")
            return ERROR_HANDLER(1423)

        # Формируем историю диалога
        all_messages = format_conversation_history(
            request_data.get("messages", []), request_data.get("new_input", "")
        )

        # Проверка на наличие изображений в последнем сообщении
        if isinstance(user_input, list):
            logger.debug(
                f"[{request_id}] Processing message with multiple content items (text/images)"
            )
            combined_text = ""
            for i, item in enumerate(user_input):
                if "text" in item:
                    combined_text += item["text"] + "\n"
                    logger.debug(f"[{request_id}] Added text content from item {i+1}")

                if "image_url" in item:
                    if model not in vision_supported_models:
                        logger.error(
                            f"[{request_id}] Model {model} does not support images"
                        )
                        return ERROR_HANDLER(1044, model)

                    # Создаем хеш URL изображения для кэширования
                    image_key = None
                    image_url = None

                    # Извлекаем URL изображения
                    if (
                        isinstance(item["image_url"], dict)
                        and "url" in item["image_url"]
                    ):
                        image_url = item["image_url"]["url"]
                    else:
                        image_url = item["image_url"]

                    # Хешируем URL для кэша
                    if image_url:
                        image_key = hashlib.md5(image_url.encode("utf-8")).hexdigest()

                    # Проверяем кэш
                    if image_key and image_key in IMAGE_CACHE:
                        cached_path = IMAGE_CACHE[image_key]
                        logger.debug(
                            f"[{request_id}] Using cached image path for item {i+1}: {cached_path}"
                        )
                        image_paths.append(cached_path)
                        image = True
                        continue

                    # Загружаем изображение, если оно не в кэше
                    logger.debug(
                        f"[{request_id}] Processing image URL in item {i+1}: {image_url[:30]}..."
                    )

                    # Загружаем изображение
                    image_path = retry_image_upload(
                        image_url, api_key, request_id=request_id
                    )

                    if image_path:
                        # Сохраняем в кэш
                        if image_key:
                            IMAGE_CACHE[image_key] = image_path
                            # Очищаем старые записи если нужно
                            if len(IMAGE_CACHE) > MAX_CACHE_SIZE:
                                old_key = next(iter(IMAGE_CACHE))
                                del IMAGE_CACHE[old_key]

                        image_paths.append(image_path)
                        image = True
                        logger.debug(
                            f"[{request_id}] Image {i+1} successfully processed: {image_path}"
                        )
                    else:
                        logger.error(f"[{request_id}] Failed to upload image {i+1}")

            # Заменяем user_input текстовой частью, только если она не пуста
            if combined_text:
                user_input = combined_text

        # Проверяем, есть ли file_ids для чата с документами
        file_ids = request_data.get("file_ids", [])
        conversation_id = request_data.get("conversation_id", None)

        # Извлекаем текст запроса для анализа ключевых слов
        prompt_text = all_messages.lower()
        extracted_prompt = messages[-1].get("content", "")
        if isinstance(extracted_prompt, list):
            extracted_prompt = " ".join([item.get("text", "") for item in extracted_prompt if "text" in item])
        extracted_prompt = extracted_prompt.lower()

        logger.debug(f"[{request_id}] Extracted prompt text: {extracted_prompt}")

        # Проверяем запрос на удаление файлов
        delete_keywords = ["удалить", "удали", "удаление", "очисти", "очистка", "delete", "remove", "clean"]
        file_keywords = ["файл", "файлы", "file", "files", "документ", "документы", "document", "documents"]
        mime_type_keywords = ["pdf", "txt", "doc", "docx", "csv", "xls", "xlsx", "json", "md", "html", "htm", "xml", "pptx", "ppt", "rtf"]

        # Объединяем все ключевые слова для файлов
        all_file_keywords = file_keywords + mime_type_keywords

        # Проверяем запрос на удаление файлов (должны быть и ключевые слова удаления, и файловые ключевые слова)
        has_delete_keywords = any(keyword in extracted_prompt for keyword in delete_keywords)
        has_file_keywords = any(keyword in extracted_prompt for keyword in all_file_keywords)

        if has_delete_keywords and has_file_keywords and user_file_ids:
            logger.info(f"[{request_id}] Deletion request detected, removing all user files")
            
            # Пытаемся получить ID команды
            team_id = None
            try:
                # Попытка получить ID команды через API
                teams_url = f"{ONE_MIN_API_URL}/teams"
                teams_headers = {"API-KEY": api_key}
                teams_response = api_request("GET", teams_url, headers=teams_headers)
                if teams_response.status_code == 200:
                    teams_data = teams_response.json()
                    if "data" in teams_data and teams_data["data"]:
                        team_id = teams_data["data"][0].get("id")
                        logger.debug(f"[{request_id}] Found team ID for deletion: {team_id}")
            except Exception as e:
                logger.error(f"[{request_id}] Error getting team ID for deletion: {str(e)}")
            
            deleted_files = []
            for file_id in user_file_ids:
                try:
                    # Формируем URL для удаления файла в зависимости от наличия team_id
                    if team_id:
                        delete_url = f"{ONE_MIN_API_URL}/teams/{team_id}/assets/{file_id}"
                    else:
                        delete_url = f"{ONE_MIN_ASSET_URL}/{file_id}"
                        
                    logger.debug(f"[{request_id}] Using URL for deletion: {delete_url}")
                    headers = {"API-KEY": api_key}
                    
                    delete_response = api_request("DELETE", delete_url, headers=headers)
                    
                    if delete_response.status_code == 200:
                        logger.info(f"[{request_id}] Successfully deleted file: {file_id}")
                        deleted_files.append(file_id)
                    else:
                        logger.error(f"[{request_id}] Failed to delete file {file_id}: {delete_response.status_code}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error deleting file {file_id}: {str(e)}")
            
            # Очищаем списох файлов пользователя в memcached
            if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None and deleted_files:
                try:
                    user_key = f"user:{api_key}"
                    safe_memcached_operation('set', user_key, json.dumps([]))
                    logger.info(f"[{request_id}] Cleared user files list in memcached")
                except Exception as e:
                    logger.error(f"[{request_id}] Error clearing user files in memcached: {str(e)}")
            
            # Отправляем ответ о удалении файлов
            return jsonify({
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Удалено файлов: {len(deleted_files)}. Список файлов очищен."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": calculate_token(prompt_text),
                    "completion_tokens": 20,
                    "total_tokens": calculate_token(prompt_text) + 20
                }
            }), 200

        # Проверяем запрос на наличие ключевых слов для обработки файлов
        has_file_reference = any(keyword in extracted_prompt for keyword in all_file_keywords)

        # Если есть file_ids и запрос содержит ключевые слова о файлах или есть ID беседы, используем CHAT_WITH_PDF
        if file_ids and len(file_ids) > 0:
            logger.debug(
                f"[{request_id}] Creating CHAT_WITH_PDF request with {len(file_ids)} files"
            )

            # Добавляем инструкцию для работы с документами к промпту
            enhanced_prompt = all_messages
            if not enhanced_prompt.strip().startswith(DOCUMENT_ANALYSIS_INSTRUCTION):
                enhanced_prompt = f"{DOCUMENT_ANALYSIS_INSTRUCTION}\n\n{all_messages}"

            # Получаем team_id пользователя
            team_id = None
            try:
                teams_url = "https://api.1min.ai/api/teams"  # Правильный URL с /api/
                teams_headers = {"API-KEY": api_key, "Content-Type": "application/json"}
                
                logger.debug(f"[{request_id}] Fetching team ID from: {teams_url}")
                teams_response = requests.get(teams_url, headers=teams_headers)
                
                if teams_response.status_code == 200:
                    teams_data = teams_response.json()
                    if "data" in teams_data and teams_data["data"]:
                        team_id = teams_data["data"][0].get("id")
                        logger.debug(f"[{request_id}] Got team ID: {team_id}")
                else:
                    logger.warning(f"[{request_id}] Failed to get team ID: {teams_response.status_code} - {teams_response.text}")
            except Exception as e:
                logger.error(f"[{request_id}] Error getting team ID: {str(e)}")

            # Если нет conversation_id, создаем новую беседу
            if not conversation_id:
                conversation_id = create_conversation_with_files(
                    file_ids, "Chat with documents", model, api_key, request_id
                )
                if not conversation_id:
                    return (
                        jsonify({"error": "Failed to create conversation with files"}),
                        500,
                    )

            # Формируем payload для запроса с файлами
            payload = {"message": enhanced_prompt}
            if conversation_id:
                payload["conversationId"] = conversation_id

            # Используем правильный URL API с /api/
            api_url = "https://api.1min.ai/api/features/conversations/messages"
            # Добавляем conversationId как параметр запроса
            api_params = {"conversationId": conversation_id}
            
            logger.debug(f"[{request_id}] Sending message to conversation using URL: {api_url} with params: {api_params}")
            
            headers = {"API-KEY": api_key, "Content-Type": "application/json"}

            # В зависимости от параметра stream выбираем способ запроса
            if stream:
                # Потоковый запрос
                return streaming_request(
                    api_url, payload, headers, request_id, model, model_settings, api_params=api_params
                )
            else:
                # Обычный запрос
                try:
                    response = requests.post(api_url, json=payload, headers=headers, params=api_params)
                    
                    logger.debug(f"[{request_id}] API response status code: {response.status_code}")
                    if response.status_code != 200:
                        logger.error(
                            f"[{request_id}] API error: {response.status_code} - {response.text}"
                        )
                        return (
                            jsonify({"error": "API request failed", "details": response.text}),
                            response.status_code,
                        )

                    # Преобразуем ответ в формат OpenAI
                    response_data = response.json()
                    logger.debug(f"[{request_id}] Raw API response: {json.dumps(response_data)[:500]}...")
                    
                    # Извлекаем ответ из разных мест структуры данных
                    ai_response = None
                    if "answer" in response_data:
                        ai_response = response_data["answer"]
                    elif "message" in response_data:
                        ai_response = response_data["message"]
                    elif "result" in response_data:
                        ai_response = response_data["result"]
                    elif "aiRecord" in response_data and "aiRecordDetail" in response_data["aiRecord"]:
                        ai_response = response_data["aiRecord"]["aiRecordDetail"].get("answer", "")
                    
                    if not ai_response:
                        # Рекурсивно ищем ответ по ключам answer, message, result
                        def find_response(obj, path=""):
                            if isinstance(obj, dict):
                                for key in ["answer", "message", "result"]:
                                    if key in obj:
                                        logger.debug(f"[{request_id}] Found response at path '{path}.{key}'")
                                        return obj[key]
                                
                                for key, value in obj.items():
                                    result = find_response(value, f"{path}.{key}")
                                    if result:
                                        return result
                            elif isinstance(obj, list):
                                for i, item in enumerate(obj):
                                    result = find_response(item, f"{path}[{i}]")
                                    if result:
                                        return result
                            return None
                        
                        ai_response = find_response(response_data)
                    
                    if not ai_response:
                        logger.error(f"[{request_id}] Could not extract AI response from API response")
                        return jsonify({"error": "Could not extract AI response"}), 500
                        
                    openai_response = format_openai_response(
                        ai_response, model, request_id
                    )
                    return jsonify(openai_response)
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Exception while processing API response: {str(e)}"
                    )
                    traceback.print_exc()
                    return jsonify({"error": str(e)}), 500

        # Подсчет токенов
        prompt_token = calculate_token(str(all_messages))

        # Проверка модели
        if PERMIT_MODELS_FROM_SUBSET_ONLY and model not in AVAILABLE_MODELS:
            return ERROR_HANDLER(1002, model)

        logger.debug(
            f"[{request_id}] Processing {prompt_token} prompt tokens with model {model}"
        )

        # Подготавливаем payload с учетом возможностей модели
        payload = prepare_payload(
            request_data, model, all_messages, image_paths, request_id
        )

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}

        # Выполнение запроса в зависимости от stream
        if not request_data.get("stream", False):
            # Обычный запрос
            logger.debug(
                f"[{request_id}] Sending non-streaming request to {ONE_MIN_API_URL}"
            )

            try:
                response = api_request(
                    "POST", ONE_MIN_API_URL, json=payload, headers=headers
                )
                logger.debug(
                    f"[{request_id}] Response status code: {response.status_code}"
                )

                if response.status_code != 200:
                    if response.status_code == 401:
                        return ERROR_HANDLER(1020, key=api_key)
                    try:
                        error_content = response.json()
                        logger.error(f"[{request_id}] Error response: {error_content}")
                    except:
                        logger.error(
                            f"[{request_id}] Could not parse error response as JSON"
                        )
                    return ERROR_HANDLER(response.status_code)

                one_min_response = response.json()
                transformed_response = transform_response(
                    one_min_response, request_data, prompt_token
                )

                response = make_response(jsonify(transformed_response))
                set_response_headers(response)
                return response, 200
            except Exception as e:
                logger.error(f"[{request_id}] Exception during request: {str(e)}")
                return jsonify({"error": str(e)}), 500
        else:
            # Потоковый запрос
            logger.debug(f"[{request_id}] Sending streaming request")

            # URL для потокового режима
            streaming_url = f"{ONE_MIN_API_URL}?isStreaming=true"

            logger.debug(f"[{request_id}] Streaming URL: {streaming_url}")
            logger.debug(f"[{request_id}] Payload: {json.dumps(payload)[:200]}...")

            try:
                # Используем сессию для управления соединением
                session = create_session()
                response_stream = session.post(
                    streaming_url, json=payload, headers=headers, stream=True
                )

                logger.debug(
                    f"[{request_id}] Streaming response status code: {response_stream.status_code}"
                )

                if response_stream.status_code != 200:
                    if response_stream.status_code == 401:
                        session.close()
                        return ERROR_HANDLER(1020, key=api_key)

                    logger.error(
                        f"[{request_id}] Error status code: {response_stream.status_code}"
                    )
                    try:
                        error_content = response_stream.json()
                        logger.error(f"[{request_id}] Error response: {error_content}")
                    except:
                        logger.error(
                            f"[{request_id}] Could not parse error response as JSON"
                        )

                    session.close()
                    return ERROR_HANDLER(response_stream.status_code)

                # Передаем сессию в generator
                return Response(
                    stream_response(
                        response_stream, request_data, model, prompt_token, session
                    ),
                    content_type="text/event-stream",
                )
            except Exception as e:
                logger.error(
                    f"[{request_id}] Exception during streaming request: {str(e)}"
                )
                return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(
            f"[{request_id}] Exception during conversation processing: {str(e)}"
        )
        traceback.print_exc()
        return (
            jsonify({"error": f"Error during conversation processing: {str(e)}"}),
            500,
        )


def parse_aspect_ratio(prompt, model, request_data, request_id=None):
    """
    Извлекает соотношение сторон из запроса или промпта и проверяет его валидность
    
    Args:
        prompt (str): Текст запроса
        model (str): Имя модели генерации изображения
        request_data (dict): Данные запроса
        request_id (str, optional): ID запроса для логирования
        
    Returns:
        tuple: (модифицированный промпт, соотношение сторон, размер изображения, сообщение об ошибке)
    """
    # Значения по умолчанию
    aspect_ratio = None
    size = request_data.get("size", "1024x1024")
    ar_error = None
    
    # Пытаемся извлечь соотношение сторон из промпта
    ar_match = re.search(r'(--|\u2014)ar\s+(\d+):(\d+)', prompt)
    if ar_match:
        width = int(ar_match.group(2))
        height = int(ar_match.group(3))
        
        # Проверяем, что соотношение не превышает 2:1 или 1:2
        if max(width, height) / min(width, height) > 2:
            ar_error = "Aspect ratio cannot exceed 2:1 or 1:2"
            logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error
        
        # Проверяем, что значения в допустимом диапазоне
        if width < 1 or width > 10000 or height < 1 or height > 10000:
            ar_error = "Aspect ratio values must be between 1 and 10000"
            logger.error(f"[{request_id}] Invalid aspect ratio values: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error
        
        # Устанавливаем соотношение сторон
        aspect_ratio = f"{width}:{height}"
        
        # Удаляем параметр из промпта
        prompt = re.sub(r'(--|\u2014)ar\s+\d+:\d+\s*', '', prompt).strip()
        
        logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_ratio}")
    
    # Если соотношения нет в промпте, проверяем в запросе
    elif "aspect_ratio" in request_data:
        aspect_ratio = request_data.get("aspect_ratio")
        
        # Проверяем, что соотношение в правильном формате
        if not re.match(r'^\d+:\d+$', aspect_ratio):
            ar_error = "Aspect ratio must be in format width:height"
            logger.error(f"[{request_id}] Invalid aspect ratio format: {aspect_ratio} - {ar_error}")
            return prompt, None, size, ar_error
        
        width, height = map(int, aspect_ratio.split(':'))
        
        # Проверяем, что соотношение не превышает 2:1 или 1:2
        if max(width, height) / min(width, height) > 2:
            ar_error = "Aspect ratio cannot exceed 2:1 or 1:2"
            logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error
        
        # Проверяем, что значения в допустимом диапазоне
        if width < 1 or width > 10000 or height < 1 or height > 10000:
            ar_error = "Aspect ratio values must be between 1 and 10000"
            logger.error(f"[{request_id}] Invalid aspect ratio values: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error
            
        logger.debug(f"[{request_id}] Using aspect ratio from request: {aspect_ratio}")
    
    # Для моделей DALL-E 3 устанавливаем соответствующие размеры
    if model == "dall-e-3" and aspect_ratio:
        width, height = map(int, aspect_ratio.split(':'))
        
        # Округляем до ближайшего допустимого соотношения для DALL-E 3
        if abs(width/height - 1) < 0.1:  # квадрат
            size = "1024x1024"
            aspect_ratio = "square"
        elif width > height:  # альбомная ориентация
            size = "1792x1024"
            aspect_ratio = "landscape"
        else:  # портретная ориентация
            size = "1024x1792"
            aspect_ratio = "portrait"
            
        logger.debug(f"[{request_id}] Adjusted size for DALL-E 3: {size}, aspect_ratio: {aspect_ratio}")
    
    return prompt, aspect_ratio, size, ar_error


@app.route("/v1/images/generations", methods=["POST", "OPTIONS"])
@limiter.limit("500 per minute")
def generate_image():
    """
    Маршрут для генерации изображений
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    # Создаем уникальный ID для запроса
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received request: /v1/images/generations")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]
    headers = {"API-KEY": api_key, "Content-Type": "application/json"}

    # Проверка, что данные переданы в правильном формате
    if request.is_json:
        request_data = request.get_json()
    else:
        logger.error(f"[{request_id}] Request content-type is not application/json")
        return jsonify({"error": "Content-type must be application/json"}), 400

    # Получаем необходимые параметры из запроса
    model = request_data.get("model", "dall-e-3").strip()
    prompt = request_data.get("prompt", "").strip()
    
    # Определим наличие негативного промпта (если есть)
    negative_prompt = None
    no_match = re.search(r'(--|\u2014)no\s+(.*?)(?=(--|\u2014)|$)', prompt)
    if no_match:
        negative_prompt = no_match.group(2).strip()
        # Удаляем негативный промпт из основного текста
        prompt = re.sub(r'(--|\u2014)no\s+.*?(?=(--|\u2014)|$)', '', prompt).strip()

    # Обрабатываем соотношение сторон и размер
    prompt, aspect_ratio, size, ar_error = parse_aspect_ratio(prompt, model, request_data, request_id)
    
    # Если была ошибка в обработке соотношения сторон, возвращаем её пользователю
    if ar_error:
        return jsonify({"error": ar_error}), 400

    # Проверка наличия промпта
    if not prompt:
        # Проверяем, есть ли промпт в сообщениях
        messages = request_data.get("messages", [])
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if last_message.get("role") == "user":
                content = last_message.get("content", "")
                if isinstance(content, str):
                    prompt = content
                elif isinstance(content, list):
                    # Собираем все текстовые части содержимого
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                    prompt = " ".join(text_parts)
        
        if prompt:
            logger.debug(f"[{request_id}] Found prompt in messages: {prompt[:100]}..." if len(prompt) > 100 else f"[{request_id}] Found prompt in messages: {prompt}")
        else:
            logger.error(f"[{request_id}] No prompt provided")
            return jsonify({"error": "A prompt is required to generate an image"}), 400

    logger.info(f"[{request_id}] Using model: {model}, prompt: '{prompt}'")

    try:
        # Определяем URL для разных моделей
        api_url = f"{ONE_MIN_API_URL}"
        
        # Таймаут 15 минут для всех моделей генерации изображений
        timeout = MIDJOURNEY_TIMEOUT
        
        # Формируем payload для запроса в зависимости от модели
        if model == "dall-e-3":
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "dall-e-3",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size or request_data.get("size", "1024x1024"),
                    "quality": request_data.get("quality", "standard"),
                    "style": request_data.get("style", "vivid"),
                },
            }
        elif model == "dall-e-2":
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "dall-e-2",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size or request_data.get("size", "1024x1024"),
                },
            }
        elif model == "stable-diffusion-xl-1024-v1-0":
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "stable-diffusion-xl-1024-v1-0",
                "promptObject": {
                    "prompt": prompt,
                    "samples": request_data.get("n", 1),
                    "size": size or request_data.get("size", "1024x1024"),
                    "cfg_scale": request_data.get("cfg_scale", 7),
                    "clip_guidance_preset": request_data.get(
                        "clip_guidance_preset", "NONE"
                    ),
                    "seed": request_data.get("seed", 0),
                    "steps": request_data.get("steps", 30),
                },
            }
        elif model == "stable-diffusion-v1-6":
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "stable-diffusion-v1-6",
                "promptObject": {
                    "prompt": prompt,
                    "samples": request_data.get("n", 1),
                    "cfg_scale": request_data.get("cfg_scale", 7),
                    "clip_guidance_preset": request_data.get(
                        "clip_guidance_preset", "NONE"
                    ),
                    "height": request_data.get("height", 512),
                    "width": request_data.get("width", 512),
                    "seed": request_data.get("seed", 0),
                    "steps": request_data.get("steps", 30),
                },
            }
        elif model in ["midjourney", "midjourney_6_1"]:
            # Допустимые соотношения сторон для Midjourney
            
            # Значения по умолчанию
            aspect_width = 1
            aspect_height = 1
            no_param = ""
            
            # Если указано соотношение сторон
            if aspect_ratio:
                # Разбиваем соотношение сторон на width и height
                ar_parts = aspect_ratio.split(":")
                aspect_width = int(ar_parts[0])
                aspect_height = int(ar_parts[1])
            
            model_name = "midjourney" if model == "midjourney" else "midjourney_6_1"
            
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": model_name,
                "promptObject": {
                    "prompt": prompt,
                    "mode": request_data.get("mode", "relax"),
                    "n": 4,  # Midjourney всегда генерирует 4 изображения
                    "aspect_width": aspect_width,
                    "aspect_height": aspect_height,
                    "isNiji6": request_data.get("isNiji6", False),
                    "maintainModeration": request_data.get("maintainModeration", True),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                    "no": request_data.get("no", no_param),
                    "image_weight": request_data.get("image_weight", 1),
                    "weird": request_data.get("weird", 0),
                },
            }
            # Если не задан negativePrompt или no, удаляем эти поля
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
            if not payload["promptObject"]["no"]:
                del payload["promptObject"]["no"]
        elif model in ["black-forest-labs/flux-schnell", "flux-schnell"]:
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "black-forest-labs/flux-schnell",
                "promptObject": {
                    "prompt": prompt,
                    "num_outputs": request_data.get("n", 1),
                    "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                    "output_format": request_data.get("output_format", "webp"),
                },
            }
        elif model in ["black-forest-labs/flux-dev", "flux-dev"]:
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "black-forest-labs/flux-dev",
                "promptObject": {
                    "prompt": prompt,
                    "num_outputs": request_data.get("n", 1),
                    "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                    "output_format": request_data.get("output_format", "webp"),
                },
            }
        elif model in ["black-forest-labs/flux-pro", "flux-pro"]:
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "black-forest-labs/flux-pro",
                "promptObject": {
                    "prompt": prompt,
                    "num_outputs": request_data.get("n", 1),
                    "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                    "output_format": request_data.get("output_format", "webp"),
                },
            }
        elif model in ["black-forest-labs/flux-1.1-pro", "flux-1.1-pro"]:
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "black-forest-labs/flux-1.1-pro",
                "promptObject": {
                    "prompt": prompt,
                    "num_outputs": request_data.get("n", 1),
                    "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                    "output_format": request_data.get("output_format", "webp"),
                },
            }
        elif model in [
            "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
            "phoenix",
        ]:  # Leonardo.ai - Phoenix
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
        elif model in [
            "b24e16ff-06e3-43eb-8d33-4416c2d75876",
            "lightning-xl",
        ]:  # Leonardo.ai - Lightning XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "b24e16ff-06e3-43eb-8d33-4416c2d75876",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
        elif model in [
            "5c232a9e-9061-4777-980a-ddc8e65647c6",
            "vision-xl",
        ]:  # Leonardo.ai - Vision XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "5c232a9e-9061-4777-980a-ddc8e65647c6",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
        elif model in [
            "e71a1c2f-4f80-4800-934f-2c68979d8cc8",
            "anime-xl",
        ]:  # Leonardo.ai - Anime XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "e71a1c2f-4f80-4800-934f-2c68979d8cc8",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
        elif model in [
            "1e60896f-3c26-4296-8ecc-53e2afecc132",
            "diffusion-xl",
        ]:  # Leonardo.ai - Diffusion XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "1e60896f-3c26-4296-8ecc-53e2afecc132",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
        elif model in [
            "aa77f04e-3eec-4034-9c07-d0f619684628",
            "kino-xl",
        ]:  # Leonardo.ai - Kino XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "aa77f04e-3eec-4034-9c07-d0f619684628",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
        elif model in [
            "2067ae52-33fd-4a82-bb92-c2c55e7d2786",
            "albedo-base-xl",
        ]:  # Leonardo.ai - Albedo Base XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "2067ae52-33fd-4a82-bb92-c2c55e7d2786",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 1),
                    "size": size or request_data.get("size", "512x512"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
        else:
            logger.error(f"[{request_id}] Invalid model: {model}")
            return ERROR_HANDLER(1002, model)

        logger.debug(f"[{request_id}] Sending request to 1min.ai API: {api_url}")
        logger.debug(f"[{request_id}] Payload: {json.dumps(payload)[:500]}")

        # Задаем параметры для повторных попыток
        max_retries = 3
        retry_count = 0
        retry_delay = 5  # Начинаем с 5 секунд между повторами

        # Для моделей Midjourney и Leonardo не делаем повторных запросов, так как они выполняются дольше
        if model.startswith("midjourney") or model.startswith("dall-e") or model.startswith("flux") or model.startswith("stable-diffusion") or model in [
            "6b645e3a-d64f-4341-a6d8-7a3690fbf042", "phoenix",  # Leonardo.ai - Phoenix
            "b24e16ff-06e3-43eb-8d33-4416c2d75876", "lightning-xl",  # Leonardo.ai - Lightning XL
            "5c232a9e-9061-4777-980a-ddc8e65647c6", "vision-xl",  # Leonardo.ai - Vision XL
            "e71a1c2f-4f80-4800-934f-2c68979d8cc8", "anime-xl",  # Leonardo.ai - Anime XL
            "1e60896f-3c26-4296-8ecc-53e2afecc132", "diffusion-xl",  # Leonardo.ai - Diffusion XL
            "aa77f04e-3eec-4034-9c07-d0f619684628", "kino-xl",  # Leonardo.ai - Kino XL
            "2067ae52-33fd-4a82-bb92-c2c55e7d2786", "albedo-base-xl"  # Leonardo.ai - Albedo Base XL
        ]:
            max_retries = 1  # Для этих моделей делаем только одну попытку

        while retry_count < max_retries:
            try:
                # Отправляем запрос с таймаутом
                response = api_request(
                    "POST", 
                    api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=timeout,
                    stream=False
                )
                
                logger.debug(f"[{request_id}] Response status code: {response.status_code}")
                
                # Если получен успешный ответ, обрабатываем его
                if response.status_code == 200:
                    break
                
                # Для Midjourney возвращаем ошибку сразу без повторных попыток
                if model.startswith("midjourney"):
                    error_msg = "Unknown error"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = error_data["error"]
                    except:
                        pass
                    return (
                        jsonify({"error": error_msg}),
                        response.status_code,
                    )
                    
                # Если ошибка 429 (Rate Limit) или 500 (Server Error), повторяем запрос
                elif response.status_code in [429, 500, 502, 503, 504]:
                    retry_count += 1
                    logger.warning(
                        f"[{request_id}] Received {response.status_code} error, retry {retry_count}/{max_retries}"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Увеличиваем время ожидания экспоненциально
                    continue
                    
                # Для других ошибок возвращаем ответ сразу
                elif response.status_code == 401:
                    return ERROR_HANDLER(1020, key=api_key)
                else:
                    error_msg = "Unknown error"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = error_data["error"]
                    except:
                        pass
                    return (
                        jsonify({"error": error_msg}),
                        response.status_code,
                    )
            except Exception as e:
                retry_count += 1
                logger.warning(
                    f"[{request_id}] Exception during API request: {str(e)}, retry {retry_count}/{max_retries}"
                )
                time.sleep(retry_delay)
                retry_delay *= 2
                
                # Для моделей Midjourney не повторяем запросы даже при исключениях
                if model.startswith("midjourney"):
                    return jsonify({"error": f"API request error: {str(e)}"}), 500
                    
                continue
                
        # Если после всех попыток по-прежнему получаем ошибки
        if retry_count >= max_retries and (not 'response' in locals() or response.status_code != 200):
            logger.error(f"[{request_id}] Max retries exceeded for image generation request")
            return jsonify({"error": "Failed to generate image after multiple attempts"}), 500

        one_min_response = response.json()

        # Преобразование ответа 1min.ai в формат OpenAI
        try:
            # Получаем все URL изображений, если они доступны
            image_urls = []
            
            # Проверяем, содержит ли ответ массив URL изображений
            result_object = one_min_response.get("aiRecord", {}).get("aiRecordDetail", {}).get("resultObject", [])
            
            if isinstance(result_object, list) and result_object:
                image_urls = result_object
            elif result_object and isinstance(result_object, str):
                image_urls = [result_object]
            
            # Если URL не найдены, попробуем другие пути извлечения
            if not image_urls:
                if "resultObject" in one_min_response:
                    if isinstance(one_min_response["resultObject"], list):
                        image_urls = one_min_response["resultObject"]
                    else:
                        image_urls = [one_min_response["resultObject"]]
            
            if not image_urls:
                logger.error(
                    f"[{request_id}] Could not extract image URLs from API response: {json.dumps(one_min_response)[:500]}"
                )
                return (
                    jsonify({"error": "Could not extract image URLs from API response"}),
                    500,
                )
            
            logger.debug(
                f"[{request_id}] Successfully generated {len(image_urls)} images"
            )
            
            # Формируем полные URL для всех изображений
            full_image_urls = []
            asset_host = "https://asset.1min.ai"
            
            for url in image_urls:
                if not url:
                    continue
                    
                # Проверяем, содержит ли url полный URL
                if not url.startswith("http"):
                    # Если изображение начинается с /, не добавляем еще один /
                    if url.startswith("/"):
                        full_url = f"{asset_host}{url}"
                    else:
                        full_url = f"{asset_host}/{url}"
                else:
                    full_url = url
                    
                full_image_urls.append(full_url)
            
            # Формируем ответ в формате OpenAI с командами для вариаций
            openai_data = []
            for i, url in enumerate(full_image_urls):
                # Создаем короткий идентификатор для изображения
                image_id = str(uuid.uuid4())[:8]
                
                # Добавляем команды для вариаций только если модель поддерживает вариации
                if model in IMAGE_VARIATION_MODELS:
                    variation_commands = {
                        "url": url,
                        "revised_prompt": prompt,
                        "variation_commands": {
                            "variation": f"/v{i+1} {url}",  # Команда для создания вариации с номером
                        }
                    }
                    openai_data.append(variation_commands)
                else:
                    openai_data.append({"url": url, "revised_prompt": prompt})
                
            openai_response = {
                "created": int(time.time()),
                "data": openai_data,
            }

            # Для совместимости с форматом текстовых ответов, добавляем structure_output
            structured_output = {"type": "image", "image_urls": full_image_urls}
            
            # Формируем markdown-текст с кнопками вариаций
            if len(full_image_urls) == 1:
                text_response = f"![Image]({full_image_urls[0]}) - [ /v1 {full_image_urls[0]} ]"
            else:
                text_response = "\n".join([f"![Image {i+1}]({url}) - [ /v{i+1} {url} ]" for i, url in enumerate(full_image_urls)])
                
            openai_response["choices"] = [
                {
                    "message": {
                        "role": "assistant",
                        "content": text_response,
                        "structured_output": structured_output
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]

            logger.info(f"[{request_id}] Returning {len(openai_data)} image URLs to client")
            response = make_response(jsonify(openai_response))
            set_response_headers(response)
            return response, 200
        except Exception as e:
            logger.error(
                f"[{request_id}] Error processing image generation response: {str(e)}"
            )
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(
            f"[{request_id}] Exception during image generation request: {str(e)}"
        )
        return jsonify({"error": str(e)}), 500


@app.route("/v1/images/variations", methods=["POST", "OPTIONS"])
@limiter.limit("500 per minute")
def image_variations():
    if request.method == "OPTIONS":
        return handle_options_request()

    # Создаем уникальный ID для запроса
    request_id = str(uuid.uuid4())
    logger.debug(f"[{request_id}] Processing image variation request")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Получение файла изображения
    if "image" not in request.files:
        logger.error(f"[{request_id}] No image file provided")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    model = request.form.get("model", "dall-e-2").strip()
    n = request.form.get("n", 1)
    size = request.form.get("size", "1024x1024")

    logger.debug(f"[{request_id}] Using model: {model} for image variations")
    
    # Проверяем, поддерживает ли модель вариации
    if model not in IMAGE_VARIATION_MODELS:
        error_msg = f"Model {model} does not support image variations. Supported models: {', '.join(IMAGE_VARIATION_MODELS)}"
        logger.error(f"[{request_id}] {error_msg}")
        return jsonify({"error": error_msg}), 400

    try:
        # Создаем новую сессию для загрузки изображения
        session = create_session()
        headers = {"API-KEY": api_key}

        # Загрузка изображения в 1min.ai
        files = {"asset": (image_file.filename, image_file, "image/png")}

        try:
            asset_response = session.post(
                ONE_MIN_ASSET_URL, files=files, headers=headers
            )
            logger.debug(
                f"[{request_id}] Image upload response status code: {asset_response.status_code}"
            )

            if asset_response.status_code != 200:
                session.close()
                logger.error(
                    f"[{request_id}] Failed to upload image: {asset_response.status_code} - {asset_response.text}"
                )
                return (
                    jsonify(
                        {"error": "Failed to upload image for variation processing"}
                    ),
                    asset_response.status_code,
                )

            # Извлекаем ID загруженного изображения
            asset_data = asset_response.json()
            logger.debug(f"[{request_id}] Asset upload response: {asset_data}")

            # Получаем URL или ID изображения
            image_id = None
            image_url = None

            # Ищем ID в разных местах структуры ответа
            if "id" in asset_data:
                image_id = asset_data["id"]
            elif "fileContent" in asset_data and "id" in asset_data["fileContent"]:
                image_id = asset_data["fileContent"]["id"]
            elif "fileContent" in asset_data and "uuid" in asset_data["fileContent"]:
                image_id = asset_data["fileContent"]["uuid"]
            
            # Если есть path, используем его как URL изображения
            if "fileContent" in asset_data and "path" in asset_data["fileContent"]:
                image_url = asset_data["fileContent"]["path"]
                # Добавляем хост, если путь относительный
                if not image_url.startswith("http"):
                    image_url = f"https://asset.1min.ai{image_url if image_url.startswith('/') else '/' + image_url}"

            if not image_id and not image_url:
                session.close()
                logger.error(f"[{request_id}] Failed to extract image ID from response")
                return jsonify({"error": "Failed to process uploaded image"}), 500

            # Формируем payload для вариации изображения
            payload = {
                "type": "IMAGE_VARIATION",
                "model": model,
                "promptObject": {
                    "n": int(n),
                    "size": size
                }
            }

            # Добавляем image_id или image_url в зависимости от того, что нашли
            if image_id:
                payload["promptObject"]["image_id"] = image_id
            elif image_url:
                payload["promptObject"]["image_url"] = image_url

            # Дополнительные параметры для специфических моделей
            if model in ["dall-e-2", "dall-e-3"]:
                # Для DALL-E добавляем дополнительные параметры
                pass
            elif model in ["midjourney", "midjourney_6_1"]:
                # Для Midjourney добавляем специфические параметры
                pass

            # Использование timeout для всех моделей (15 минут)
            timeout = MIDJOURNEY_TIMEOUT
                
            logger.debug(f"[{request_id}] Sending variation request with payload: {payload}")

            # Отправляем запрос на создание вариации
            variation_response = api_request(
                "POST",
                f"{ONE_MIN_API_URL}",
                headers={"API-KEY": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=timeout
            )

            if variation_response.status_code != 200:
                session.close()
                logger.error(
                    f"[{request_id}] Variation request failed: {variation_response.status_code} - {variation_response.text}"
                )
                return (
                    jsonify({"error": "Failed to create image variations"}),
                    variation_response.status_code,
                )

            # Обрабатываем ответ и формируем результат
            variation_data = variation_response.json()
            logger.debug(f"[{request_id}] Variation response: {variation_data}")

            # Извлекаем URL вариаций
            variation_urls = []
            
            # Пытаемся найти URL вариаций в ответе
            if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                record_detail = variation_data["aiRecord"]["aiRecordDetail"]
                if "resultObject" in record_detail:
                    result = record_detail["resultObject"]
                    if isinstance(result, list):
                        variation_urls = result
                    elif isinstance(result, str):
                        variation_urls = [result]
            
            # Альтернативный путь поиска
            if not variation_urls and "resultObject" in variation_data:
                result = variation_data["resultObject"]
                if isinstance(result, list):
                    variation_urls = result
                elif isinstance(result, str):
                    variation_urls = [result]

            if not variation_urls:
                session.close()
                logger.error(f"[{request_id}] No variation URLs found in response")
                return jsonify({"error": "No variations were generated"}), 500

            # Формируем полные URL для вариаций
            full_variation_urls = []
            asset_host = "https://asset.1min.ai"
            
            for url in variation_urls:
                if not url:
                    continue
                    
                # Если URL не полный, добавляем хост
                if not url.startswith("http"):
                    if url.startswith("/"):
                        full_url = f"{asset_host}{url}"
                    else:
                        full_url = f"{asset_host}/{url}"
                else:
                    full_url = url
                    
                full_variation_urls.append(full_url)

            # Формируем ответ в формате OpenAI
            openai_data = []
            for url in full_variation_urls:
                openai_data.append({"url": url})

            openai_response = {
                "created": int(time.time()),
                "data": openai_data,
            }

            # Добавляем текст с кнопками вариаций для markdown-отображения
            markdown_text = ""
            if len(full_variation_urls) == 1:
                markdown_text = f"![Variation]({full_variation_urls[0]}) - [ /v1 {full_variation_urls[0]} ]"
            else:
                markdown_text = "\n".join([f"![Variation {i+1}]({url}) - [ /v{i+1} {url} ]" for i, url in enumerate(full_variation_urls)])
            
            # Добавляем текстовый контент в ответ
            openai_response["choices"] = [
                {
                    "message": {
                        "role": "assistant",
                        "content": markdown_text
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]

            session.close()
            logger.info(f"[{request_id}] Successfully generated {len(openai_data)} image variations")
            return jsonify(openai_response), 200

        except Exception as e:
            session.close()
            logger.error(f"[{request_id}] Exception during variation request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Exception during image variation processing: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/assistants", methods=["POST", "OPTIONS"])
@limiter.limit("500 per minute")
def create_assistant():
    if request.method == "OPTIONS":
        return handle_options_request()

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]
    headers = {"API-KEY": api_key, "Content-Type": "application/json"}

    request_data = request.json
    name = request_data.get("name", "PDF Assistant")
    instructions = request_data.get("instructions", "")
    model = request_data.get("model", "gpt-4o-mini")
    file_ids = request_data.get("file_ids", [])

    # Создание беседы с PDF в 1min.ai
    payload = {
        "title": name,
        "type": "CHAT_WITH_PDF",
        "model": model,
        "fileList": file_ids,
    }

    response = requests.post(
        ONE_MIN_CONVERSATION_API_URL, json=payload, headers=headers
    )

    if response.status_code != 200:
        if response.status_code == 401:
            return ERROR_HANDLER(1020, key=api_key)
        return (
            jsonify({"error": response.json().get("error", "Unknown error")}),
            response.status_code,
        )

    one_min_response = response.json()

    try:
        conversation_id = one_min_response.get("id")

        openai_response = {
            "id": f"asst_{conversation_id}",
            "object": "assistant",
            "created_at": int(time.time()),
            "name": name,
            "description": None,
            "model": model,
            "instructions": instructions,
            "tools": [],
            "file_ids": file_ids,
            "metadata": {},
        }

        response = make_response(jsonify(openai_response))
        set_response_headers(response)
        return response, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def handle_options_request():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response, 204


def transform_response(one_min_response, request_data, prompt_token):
    try:
        # Вывод структуры ответа для отладки
        logger.debug(f"Response structure: {json.dumps(one_min_response)[:200]}...")

        # Получаем ответ из соответствующего места в JSON
        result_text = (
            one_min_response.get("aiRecord", {})
            .get("aiRecordDetail", {})
            .get("resultObject", [""])[0]
        )

        if not result_text:
            # Альтернативные пути извлечения ответа
            if "resultObject" in one_min_response:
                result_text = (
                    one_min_response["resultObject"][0]
                    if isinstance(one_min_response["resultObject"], list)
                    else one_min_response["resultObject"]
                )
            elif "result" in one_min_response:
                result_text = one_min_response["result"]
            else:
                # Если не нашли ответ по известным путям, возвращаем ошибку
                logger.error(f"Cannot extract response text from API result")
                result_text = "Error: Could not extract response from API"

        completion_token = calculate_token(result_text)
        logger.debug(
            f"Finished processing Non-Streaming response. Completion tokens: {str(completion_token)}"
        )
        logger.debug(f"Total tokens: {str(completion_token + prompt_token)}")

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "mistral-nemo").strip(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token,
                "completion_tokens": completion_token,
                "total_tokens": prompt_token + completion_token,
            },
        }
    except Exception as e:
        logger.error(f"Error in transform_response: {str(e)}")
        # Возвращаем ошибку в формате, совместимом с OpenAI
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "mistral-nemo").strip(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Error processing response: {str(e)}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token,
                "completion_tokens": 0,
                "total_tokens": prompt_token,
            },
        }


def set_response_headers(response):
    response.headers["Content-Type"] = "application/json"
    response.headers["Access-Control-Allow-Origin"] = "*"  # Исправил дефис в имени заголовка
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    # Добавляем больше CORS заголовков
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
    return response  # Возвращаем ответ для цепочки


def stream_response(response, request_data, model, prompt_tokens, session=None):
    """
    Stream полученный от 1min.ai ответ в формате, совместимом с OpenAI API.
    """
    all_chunks = ""
    
    # Отправляем первый фрагмент: роль сообщения
    first_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    
    yield f"data: {json.dumps(first_chunk)}\n\n"
    
    # Более простая реализация из main (6).py для обработки контента
    for chunk in response.iter_content(chunk_size=1024):
        finish_reason = None

        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk.decode('utf-8')
                    },
                    "finish_reason": finish_reason
                }
            ]
        }
        all_chunks += chunk.decode('utf-8')
        yield f"data: {json.dumps(return_chunk)}\n\n"
    
    tokens = calculate_token(all_chunks)
    logger.debug(f"Finished processing streaming response. Completion tokens: {str(tokens)}")
    logger.debug(f"Total tokens: {str(tokens + prompt_tokens)}")
    
    # Финальный чанк, обозначающий конец потока
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": ""    
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens + prompt_tokens
        }
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


def safe_temp_file(prefix, request_id=None):
    """
    Безопасно создает временный файл и гарантирует его удаление после использования

    Args:
        prefix: Префикс для имени файла
        request_id: ID запроса для логирования

    Returns:
        str: Путь к временному файлу
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

    # Создаем временную директорию, если её нет
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Очищаем старые файлы (старше 1 часа)
    try:
        current_time = time.time()
        for old_file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, old_file)
            if os.path.isfile(file_path):
                # Если файл старше 1 часа - удаляем
                if current_time - os.path.getmtime(file_path) > 3600:
                    try:
                        os.remove(file_path)
                        logger.debug(
                            f"[{request_id}] Removed old temp file: {file_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[{request_id}] Failed to remove old temp file {file_path}: {str(e)}"
                        )
    except Exception as e:
        logger.warning(f"[{request_id}] Error while cleaning old temp files: {str(e)}")

    # Создаем новый временный файл
    temp_file_path = os.path.join(temp_dir, f"{prefix}_{request_id}_{random_string}")
    return temp_file_path


def retry_image_upload(image_url, api_key, request_id=None):
    """Загружает изображение с повторными попытками, возвращает прямую ссылку на него"""
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Uploading image: {image_url}")

    # Создаем новую сессию для этого запроса
    session = create_session()
    temp_file_path = None

    try:
        # Загружаем изображение
        if image_url.startswith(("http://", "https://")):
            # Загрузка по URL
            logger.debug(f"[{request_id}] Fetching image from URL: {image_url}")
            response = session.get(image_url, stream=True)
            response.raise_for_status()
            image_data = response.content
        else:
            # Декодирование base64
            logger.debug(f"[{request_id}] Decoding base64 image")
            image_data = base64.b64decode(image_url.split(",")[1])

        # Проверяем размер файла
        if len(image_data) == 0:
            logger.error(f"[{request_id}] Empty image data")
            return None

        # Создаем временный файл
        temp_file_path = safe_temp_file("image", request_id)

        with open(temp_file_path, "wb") as f:
            f.write(image_data)

        # Проверяем, что файл не пуст
        if os.path.getsize(temp_file_path) == 0:
            logger.error(f"[{request_id}] Empty image file created: {temp_file_path}")
            return None

        # Загружаем на сервер
        try:
            with open(temp_file_path, "rb") as f:
                upload_response = session.post(
                    ONE_MIN_ASSET_URL,
                    headers={"API-KEY": api_key},
                    files={
                        "asset": (
                            os.path.basename(image_url),
                            f,
                            (
                                "image/webp"
                                if image_url.endswith(".webp")
                                else "image/jpeg"
                            ),
                        )
                    },
                )

                if upload_response.status_code != 200:
                    logger.error(
                        f"[{request_id}] Upload failed with status {upload_response.status_code}: {upload_response.text}"
                    )
                    return None

                # Получаем URL изображения
                upload_data = upload_response.json()
                if isinstance(upload_data, str):
                    try:
                        upload_data = json.loads(upload_data)
                    except:
                        logger.error(
                            f"[{request_id}] Failed to parse upload response: {upload_data}"
                        )
                        return None

                logger.debug(f"[{request_id}] Upload response: {upload_data}")

                # Получаем путь к файлу из fileContent
                if (
                    "fileContent" in upload_data
                    and "path" in upload_data["fileContent"]
                ):
                    url = upload_data["fileContent"]["path"]
                    logger.info(f"[{request_id}] Image uploaded successfully: {url}")
                    return url

                logger.error(f"[{request_id}] No path found in upload response")
                return None

        except Exception as e:
            logger.error(f"[{request_id}] Exception during image upload: {str(e)}")
            return None

    except Exception as e:
        logger.error(f"[{request_id}] Exception during image processing: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # Закрываем сессию
        session.close()
        # Удаляем временный файл
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"[{request_id}] Removed temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(
                    f"[{request_id}] Failed to remove temp file {temp_file_path}: {str(e)}"
                )


def create_session():
    """Создает новую сессию с оптимальными настройками для API-запросов"""
    session = requests.Session()

    # Настройка повторных попыток для всех запросов
    retry_strategy = requests.packages.urllib3.util.retry.Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def upload_document(file_data, file_name, api_key, request_id=None):
    """
    Загружает файл/документ на сервер и возвращает его ID.

    Args:
        file_data: бинарное содержимое файла
        file_name: имя файла
        api_key: API-ключ пользователя
        request_id: ID запроса для логирования

    Returns:
        str: ID загруженного файла или None в случае ошибки
    """
    session = create_session()
    try:
        # Определяем тип файла по расширению
        extension = os.path.splitext(file_name)[1].lower()
        logger.info(f"[{request_id}] Uploading document: {file_name}")

        # Словарь с MIME-типами для разных расширений файлов
        mime_types = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".csv": "text/csv",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".json": "application/json",
            ".md": "text/markdown",
            ".html": "text/html",
            ".htm": "text/html",
            ".xml": "application/xml",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".rtf": "application/rtf",
        }

        # Получаем MIME-тип из словаря или используем octet-stream по умолчанию
        mime_type = mime_types.get(extension, "application/octet-stream")

        # Определяем тип файла для специальной обработки
        file_type = None
        if extension in [".doc"]:
            file_type = "DOC"
        elif extension in [".docx"]:
            file_type = "DOCX"

        # Загружаем файл на сервер - добавим больше деталей в логи
        logger.info(
            f"[{request_id}] Uploading file to 1min.ai: {file_name} ({mime_type}, {len(file_data)} bytes)"
        )

        headers = {"API-KEY": api_key}

        # Специальные заголовки для DOC/DOCX
        if file_type in ["DOC", "DOCX"]:
            headers["X-File-Type"] = file_type

        files = {"asset": (file_name, file_data, mime_type)}

        upload_response = session.post(ONE_MIN_ASSET_URL, headers=headers, files=files)

        if upload_response.status_code != 200:
            logger.error(
                f"[{request_id}] Document upload failed: {upload_response.status_code} - {upload_response.text}"
            )
            return None

        # Подробное логирование ответа
        try:
            response_text = upload_response.text
            logger.debug(
                f"[{request_id}] Raw upload response: {response_text[:500]}..."
            )

            response_data = upload_response.json()
            logger.debug(
                f"[{request_id}] Upload response JSON: {json.dumps(response_data)[:500]}..."
            )

            file_id = None
            if "id" in response_data:
                file_id = response_data["id"]
                logger.debug(f"[{request_id}] Found file ID at top level: {file_id}")
            elif (
                "fileContent" in response_data and "id" in response_data["fileContent"]
            ):
                file_id = response_data["fileContent"]["id"]
                logger.debug(f"[{request_id}] Found file ID in fileContent: {file_id}")
            elif (
                "fileContent" in response_data and "uuid" in response_data["fileContent"]
            ):
                file_id = response_data["fileContent"]["uuid"]
                logger.debug(f"[{request_id}] Found file ID (uuid) in fileContent: {file_id}")
            else:
                # Пытаемся найти ID в других местах структуры ответа
                if isinstance(response_data, dict):
                    # Рекурсивный поиск id в структуре ответа
                    def find_id(obj, path="root"):
                        if isinstance(obj, dict):
                            if "id" in obj:
                                logger.debug(
                                    f"[{request_id}] Found ID at path '{path}': {obj['id']}"
                                )
                                return obj["id"]
                            if "uuid" in obj:
                                logger.debug(
                                    f"[{request_id}] Found UUID at path '{path}': {obj['uuid']}"
                                )
                                return obj["uuid"]
                            for k, v in obj.items():
                                result = find_id(v, f"{path}.{k}")
                                if result:
                                    return result
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                result = find_id(item, f"{path}[{i}]")
                                if result:
                                    return result
                        return None

                    file_id = find_id(response_data)

            if not file_id:
                logger.error(
                    f"[{request_id}] Could not find file ID in response: {json.dumps(response_data)}"
                )
                return None

            logger.info(
                f"[{request_id}] Document uploaded successfully. File ID: {file_id}"
            )
            return file_id
        except Exception as e:
            logger.error(f"[{request_id}] Error parsing upload response: {str(e)}")
            traceback.print_exc()
            return None
    except Exception as e:
        logger.error(f"[{request_id}] Error uploading document: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        session.close()


@app.route("/v1/files", methods=["POST"])
@limiter.limit("100 per minute")
def upload_file():
    """
    Маршрут для загрузки файлов (аналог OpenAI Files API)
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received file upload request")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    if "file" not in request.files:
        logger.error(f"[{request_id}] No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.error(f"[{request_id}] No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Сохраняем файл в память
        file_data = file.read()
        file_name = file.filename

        # Загружаем файл на сервер 1min.ai
        file_id = upload_document(file_data, file_name, api_key, request_id)

        if not file_id:
            return jsonify({"error": "Failed to upload file"}), 500

        # Сохраняем ID файла в сессии пользователя через memcached, если он доступен
        if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
            try:
                user_key = f"user:{api_key}"
                # Получаем текущие файлы пользователя или создаем новый список
                user_files_json = safe_memcached_operation('get', user_key)
                user_files = []
                
                if user_files_json:
                    try:
                        if isinstance(user_files_json, str):
                            user_files = json.loads(user_files_json)
                        elif isinstance(user_files_json, bytes):
                            user_files = json.loads(user_files_json.decode('utf-8'))
                    except Exception as e:
                        logger.error(f"[{request_id}] Error parsing user files from memcached: {str(e)}")
                        user_files = []
                        
                # Добавляем новый файл
                file_info = {
                    "id": file_id,
                    "filename": file_name,
                    "uploaded_at": int(time.time())
                }
                
                # Проверяем, что файла с таким ID еще нет в списке
                if not any(f.get("id") == file_id for f in user_files):
                    user_files.append(file_info)
                
                # Сохраняем обновленный список файлов
                safe_memcached_operation('set', user_key, json.dumps(user_files))
                logger.info(f"[{request_id}] Saved file ID {file_id} for user in memcached")
                
                # Добавляем пользователя в список известных пользователей для функции очистки
                known_users_list_json = safe_memcached_operation('get', 'known_users_list')
                known_users_list = []
                
                if known_users_list_json:
                    try:
                        if isinstance(known_users_list_json, str):
                            known_users_list = json.loads(known_users_list_json)
                        elif isinstance(known_users_list_json, bytes):
                            known_users_list = json.loads(known_users_list_json.decode('utf-8'))
                    except Exception as e:
                        logger.error(f"[{request_id}] Error parsing known users list: {str(e)}")
                
                # Добавляем API-ключ в список известных пользователей, если его еще нет
                if api_key not in known_users_list:
                    known_users_list.append(api_key)
                    safe_memcached_operation('set', 'known_users_list', json.dumps(known_users_list))
                    logger.debug(f"[{request_id}] Added user to known_users_list for cleanup")
            except Exception as e:
                logger.error(f"[{request_id}] Error saving file info to memcached: {str(e)}")

        # Создаем ответ в формате OpenAI API
        response_data = {
            "id": file_id,
            "object": "file",
            "bytes": len(file_data),
            "created_at": int(time.time()),
            "filename": file_name,
            "purpose": request.form.get("purpose", "assistants")
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"[{request_id}] Exception during file upload: {str(e)}")
        return jsonify({"error": str(e)}), 500


def emulate_stream_response(full_content, request_data, model, prompt_tokens):
    """
    Эмулирует потоковый ответ для случаев, когда API не поддерживает потоковую передачу

    Args:
        full_content: Полный текст ответа
        request_data: Данные запроса
        model: Модель
        prompt_tokens: Количество токенов в запросе

    Yields:
        str: строки для потоковой передачи
    """
    # Разбиваем ответ на фрагменты по ~5 слов
    words = full_content.split()
    chunks = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]

    for chunk in chunks:
        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
            ],
        }

        yield f"data: {json.dumps(return_chunk)}\n\n"
        time.sleep(0.05)  # Небольшая задержка для эмуляции потока

    # Подсчитываем токены
    tokens = calculate_token(full_content)

    # Финальный чанк
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens + prompt_tokens,
        },
    }

    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# Функция для выполнения запроса к API с новой сессией
def api_request(req_method, url, headers=None, 
                requester_ip=None, data=None, 
                files=None, stream=False, 
                timeout=None, json=None, **kwargs):
    """Выполняет HTTP-запрос к API с нормализацией URL и обработкой ошибок"""
    req_url = url.strip()
    logger.debug(f"API request URL: {req_url}")
    
    # Параметры запроса
    req_params = {}
    if headers:
        req_params["headers"] = headers
    if data:
        req_params["data"] = data
    if files:
        req_params["files"] = files
    if stream:
        req_params["stream"] = stream
    if json:
        req_params["json"] = json
    
    # Добавляем остальные параметры
    req_params.update(kwargs)
    
    # Используем увеличенный таймаут для запросов Midjourney
    is_midjourney = False
    
    # Проверяем JSON на наличие упоминаний Midjourney
    if json and isinstance(json, dict):
        model_name = json.get("model", "")
        prompt_type = json.get("type", "")
        if "midjourney" in model_name.lower() or (prompt_type == "IMAGE_GENERATOR" and "midjourney" in str(json).lower()):
            is_midjourney = True
    
    # Проверяем все параметры запроса на наличие упоминаний Midjourney
    if not is_midjourney and "midjourney" in str(req_params).lower():
        is_midjourney = True
        
    if is_midjourney:
        req_params["timeout"] = timeout or MIDJOURNEY_TIMEOUT
        logger.debug(f"Using extended timeout for Midjourney: {MIDJOURNEY_TIMEOUT}s")
    else:
        req_params["timeout"] = timeout or DEFAULT_TIMEOUT
        
    # Выполняем запрос
    try:
        response = requests.request(req_method, req_url, **req_params)
        return response
    except Exception as e:
        logger.error(f"API request error: {str(e)}")
        raise


@app.route("/v1/audio/transcriptions", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_transcriptions():
    """
    Маршрут для преобразования речи в текст (аналог OpenAI Whisper API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/audio/transcriptions")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Проверка наличия файла аудио
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
        # Создаем новую сессию для загрузки аудио
        session = create_session()
        headers = {"API-KEY": api_key}

        # Загрузка аудио в 1min.ai
        files = {"asset": (audio_file.filename, audio_file, "audio/mpeg")}

        try:
            asset_response = session.post(
                ONE_MIN_ASSET_URL, files=files, headers=headers
            )
            logger.debug(
                f"[{request_id}] Audio upload response status code: {asset_response.status_code}"
            )

            if asset_response.status_code != 200:
                session.close()
                return (
                    jsonify(
                        {
                            "error": asset_response.json().get(
                                "error", "Failed to upload audio"
                            )
                        }
                    ),
                    asset_response.status_code,
                )

            audio_path = asset_response.json()["fileContent"]["path"]
            logger.debug(f"[{request_id}] Successfully uploaded audio: {audio_path}")
        finally:
            session.close()

            # Формируем payload для запроса SPEECH_TO_TEXT
            payload = {
                "type": "SPEECH_TO_TEXT",
                "model": "whisper-1",
                "promptObject": {
                    "audioUrl": audio_path,
                    "response_format": response_format,
                },
            }

        # Добавляем дополнительные параметры, если они предоставлены
        if language:
            payload["promptObject"]["language"] = language

        if temperature is not None:
            payload["promptObject"]["temperature"] = float(temperature)

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}

        # Отправляем запрос
        logger.debug(
            f"[{request_id}] Sending transcription request to {ONE_MIN_API_URL}"
        )
        response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
        logger.debug(
            f"[{request_id}] Transcription response status code: {response.status_code}"
        )

        if response.status_code != 200:
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            logger.error(
                f"[{request_id}] Error in transcription response: {response.text[:200]}"
            )
            return (
                jsonify({"error": response.json().get("error", "Unknown error")}),
                response.status_code,
            )

        # Преобразуем ответ в формат OpenAI API
        one_min_response = response.json()

        try:
            # Извлекаем текст из ответа
            result_text = ""

            if (
                "aiRecord" in one_min_response
                and "aiRecordDetail" in one_min_response["aiRecord"]
            ):
                result_text = one_min_response["aiRecord"]["aiRecordDetail"].get(
                    "resultObject", [""]
                )[0]
            elif "resultObject" in one_min_response:
                result_text = (
                    one_min_response["resultObject"][0]
                    if isinstance(one_min_response["resultObject"], list)
                    else one_min_response["resultObject"]
                )

            # Проверяем, не является ли result_text JSON строкой
            try:
                # Если result_text - это JSON строка, распарсим ее
                if result_text and result_text.strip().startswith("{"):
                    parsed_json = json.loads(result_text)
                    # Если в parsed_json есть поле "text", используем его значение
                    if "text" in parsed_json:
                        result_text = parsed_json["text"]
                        logger.debug(f"[{request_id}] Extracted inner text from JSON: {result_text}")
            except (json.JSONDecodeError, TypeError, ValueError):
                # Если не удалось распарсить как JSON, используем как есть
                logger.debug(f"[{request_id}] Using result_text as is: {result_text}")
                pass

            if not result_text:
                logger.error(
                    f"[{request_id}] Could not extract transcription text from API response"
                )
                return jsonify({"error": "Could not extract transcription text"}), 500

            # Максимально простой и надежный формат ответа
            logger.info(f"[{request_id}] Successfully processed audio transcription: {result_text}")
            
            # Создаем JSON ответ строго в формате OpenAI API
            response_data = {"text": result_text}
            
            # Добавляем CORS заголовки
            response = jsonify(response_data)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
            
            return response

        except Exception as e:
            logger.error(
                f"[{request_id}] Error processing transcription response: {str(e)}"
            )
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Exception during transcription request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/audio/translations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_translations():
    """
    Маршрут для перевода аудио в текст (аналог OpenAI Whisper API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/audio/translations")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Проверка наличия файла аудио
    if "file" not in request.files:
        logger.error(f"[{request_id}] No audio file provided")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    model = request.form.get("model", "whisper-1")
    response_format = request.form.get("response_format", "text")
    temperature = request.form.get("temperature", 0)

    logger.info(f"[{request_id}] Processing audio translation with model {model}")

    try:
        # Создаем новую сессию для загрузки аудио
        session = create_session()
        headers = {"API-KEY": api_key}

        # Загрузка аудио в 1min.ai
        files = {"asset": (audio_file.filename, audio_file, "audio/mpeg")}

        try:
            asset_response = session.post(
                ONE_MIN_ASSET_URL, files=files, headers=headers
            )
            logger.debug(
                f"[{request_id}] Audio upload response status code: {asset_response.status_code}"
            )

            if asset_response.status_code != 200:
                session.close()
                return (
                    jsonify(
                        {
                            "error": asset_response.json().get(
                                "error", "Failed to upload audio"
                            )
                        }
                    ),
                    asset_response.status_code,
                )

            audio_path = asset_response.json()["fileContent"]["path"]
            logger.debug(f"[{request_id}] Successfully uploaded audio: {audio_path}")
        finally:
            session.close()

            # Формируем payload для запроса AUDIO_TRANSLATOR
            payload = {
                "type": "AUDIO_TRANSLATOR",
                "model": "whisper-1",
                "promptObject": {
                    "audioUrl": audio_path,
                    "response_format": response_format,
                    "temperature": float(temperature),
                },
            }

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}

        # Отправляем запрос
        logger.debug(f"[{request_id}] Sending translation request to {ONE_MIN_API_URL}")
        response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
        logger.debug(
            f"[{request_id}] Translation response status code: {response.status_code}"
        )

        if response.status_code != 200:
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            logger.error(
                f"[{request_id}] Error in translation response: {response.text[:200]}"
            )
            return (
                jsonify({"error": response.json().get("error", "Unknown error")}),
                response.status_code,
            )

        # Преобразуем ответ в формат OpenAI API
        one_min_response = response.json()

        try:
            # Извлекаем текст из ответа
            result_text = ""

            if (
                "aiRecord" in one_min_response
                and "aiRecordDetail" in one_min_response["aiRecord"]
            ):
                result_text = one_min_response["aiRecord"]["aiRecordDetail"].get(
                    "resultObject", [""]
                )[0]
            elif "resultObject" in one_min_response:
                result_text = (
                    one_min_response["resultObject"][0]
                    if isinstance(one_min_response["resultObject"], list)
                    else one_min_response["resultObject"]
                )

            if not result_text:
                logger.error(
                    f"[{request_id}] Could not extract translation text from API response"
                )
                return jsonify({"error": "Could not extract translation text"}), 500

            # Максимально простой и надежный формат ответа
            logger.info(f"[{request_id}] Successfully processed audio translation: {result_text}")
            
            # Создаем JSON ответ строго в формате OpenAI API
            response_data = {"text": result_text}
            
            # Добавляем CORS заголовки
            response = jsonify(response_data)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
            
            return response

        except Exception as e:
            logger.error(
                f"[{request_id}] Error processing translation response: {str(e)}"
            )
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Exception during translation request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/audio/speech", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def text_to_speech():
    """
    Маршрут для преобразования текста в речь (аналог OpenAI TTS API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/audio/speech")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Получаем данные запроса
    request_data = request.json
    model = request_data.get("model", "tts-1")
    input_text = request_data.get("input", "")
    voice = request_data.get("voice", "alloy")
    response_format = request_data.get("response_format", "mp3")
    speed = request_data.get("speed", 1.0)

    logger.info(f"[{request_id}] Processing TTS request with model {model}")
    logger.debug(f"[{request_id}] Text input: {input_text[:100]}...")

    if not input_text:
        logger.error(f"[{request_id}] No input text provided")
        return jsonify({"error": "No input text provided"}), 400
        
    try:
        # Формируем payload для запроса TEXT_TO_SPEECH
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
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            logger.error(f"[{request_id}] Error in TTS response: {response.text[:200]}")
            return (
                jsonify({"error": response.json().get("error", "Unknown error")}),
                response.status_code,
            )

        # Обрабатываем ответ
        one_min_response = response.json()
        
        try:
            # Получаем URL аудио из ответа
            audio_url = ""
            
            if "aiRecord" in one_min_response and "aiRecordDetail" in one_min_response["aiRecord"]:
                result_object = one_min_response["aiRecord"]["aiRecordDetail"].get("resultObject", "")
                if isinstance(result_object, list) and result_object:
                    audio_url = result_object[0]
                else:
                    audio_url = result_object
            elif "resultObject" in one_min_response:
                result_object = one_min_response["resultObject"]
                if isinstance(result_object, list) and result_object:
                    audio_url = result_object[0]
                else:
                    audio_url = result_object
            
            if not audio_url:
                logger.error(f"[{request_id}] Could not extract audio URL from API response")
                return jsonify({"error": "Could not extract audio URL"}), 500
            
            # Получаем аудио-данные по URL
            audio_response = api_request("GET", f"https://asset.1min.ai/{audio_url}")
            
            if audio_response.status_code != 200:
                logger.error(f"[{request_id}] Failed to download audio: {audio_response.status_code}")
                return jsonify({"error": "Failed to download audio"}), 500
            
            # Возвращаем аудио клиенту
            logger.info(f"[{request_id}] Successfully generated speech audio")
            
            # Создаем ответ с аудио и правильными MIME-типами
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


# Функции для работы с файлами в API
@app.route("/v1/files", methods=["GET", "POST", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_files():
    """
    Маршрут для работы с файлами: получение списка и загрузка новых файлов
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]
    
    # GET - получение списка файлов
    if request.method == "GET":
        logger.info(f"[{request_id}] Received request: GET /v1/files")
        try:
            # Получаем список файлов пользователя из memcached
            files = []
            if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
                try:
                    user_key = f"user:{api_key}"
                    user_files_json = safe_memcached_operation('get', user_key)
                    
                    if user_files_json:
                        try:
                            if isinstance(user_files_json, str):
                                user_files = json.loads(user_files_json)
                            elif isinstance(user_files_json, bytes):
                                user_files = json.loads(user_files_json.decode('utf-8'))
                            else:
                                user_files = user_files_json
                                
                            # Преобразуем данные о файлах в формат ответа API
                            for file_info in user_files:
                                if isinstance(file_info, dict) and "id" in file_info:
                                    files.append({
                                        "id": file_info.get("id"),
                                        "object": "file",
                                        "bytes": file_info.get("bytes", 0),
                                        "created_at": file_info.get("created_at", int(time.time())),
                                        "filename": file_info.get("filename", f"file_{file_info.get('id')}"),
                                        "purpose": "assistants",
                                        "status": "processed"
                                    })
                            logger.debug(f"[{request_id}] Found {len(files)} files for user in memcached")
                        except Exception as e:
                            logger.error(f"[{request_id}] Error parsing user files from memcached: {str(e)}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")
            
            # Формируем ответ в формате OpenAI API
            response_data = {
                "data": files,
                "object": "list"
            }
            response = make_response(jsonify(response_data))
            set_response_headers(response)
            return response
        except Exception as e:
            logger.error(f"[{request_id}] Exception during files list request: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    # POST - загрузка нового файла
    elif request.method == "POST":
        logger.info(f"[{request_id}] Received request: POST /v1/files")
        
        # Проверка наличия файла
        if "file" not in request.files:
            logger.error(f"[{request_id}] No file provided")
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files["file"]
        purpose = request.form.get("purpose", "assistants")
        
        try:
            # Получаем содержимое файла
            file_data = file.read()
            file_name = file.filename
            
            # Получаем ID загруженного файла
            file_id = upload_document(file_data, file_name, api_key, request_id)
            
            if not file_id:
                logger.error(f"[{request_id}] Failed to upload file")
                return jsonify({"error": "Failed to upload file"}), 500
                
            # Формируем ответ в формате OpenAI API
            response_data = {
                "id": file_id,
                "object": "file",
                "bytes": len(file_data),
                "created_at": int(time.time()),
                "filename": file_name,
                "purpose": purpose,
                "status": "processed"
            }
            
            logger.info(f"[{request_id}] File uploaded successfully: {file_id}")
            response = make_response(jsonify(response_data))
            set_response_headers(response)
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] Exception during file upload: {str(e)}")
            return jsonify({"error": str(e)}), 500


@app.route("/v1/files/<file_id>", methods=["GET", "DELETE", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file(file_id):
    """
    Маршрут для работы с конкретным файлом: получение информации и удаление
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]
    
    # GET - получение информации о файле
    if request.method == "GET":
        logger.info(f"[{request_id}] Received request: GET /v1/files/{file_id}")
        try:
            # Ищем файл в сохраненных файлах пользователя в memcached
            file_info = None
            if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
                try:
                    user_key = f"user:{api_key}"
                    user_files_json = safe_memcached_operation('get', user_key)
                    
                    if user_files_json:
                        try:
                            if isinstance(user_files_json, str):
                                user_files = json.loads(user_files_json)
                            elif isinstance(user_files_json, bytes):
                                user_files = json.loads(user_files_json.decode('utf-8'))
                            else:
                                user_files = user_files_json
                                
                            # Ищем файл с указанным ID
                            for file_item in user_files:
                                if file_item.get("id") == file_id:
                                    file_info = file_item
                                    logger.debug(f"[{request_id}] Found file info in memcached: {file_id}")
                                    break
                        except Exception as e:
                            logger.error(f"[{request_id}] Error parsing user files from memcached: {str(e)}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")
            
            # Если файл не найден, возвращаем заполнитель
            if not file_info:
                logger.debug(f"[{request_id}] File not found in memcached, using placeholder: {file_id}")
                file_info = {
                    "id": file_id,
                    "bytes": 0,
                    "created_at": int(time.time()),
                    "filename": f"file_{file_id}"
                }
            
            # Формируем ответ в формате OpenAI API
            response_data = {
                "id": file_info.get("id"),
                "object": "file",
                "bytes": file_info.get("bytes", 0),
                "created_at": file_info.get("created_at", int(time.time())),
                "filename": file_info.get("filename", f"file_{file_id}"),
                "purpose": "assistants",
                "status": "processed"
            }
            
            response = make_response(jsonify(response_data))
            set_response_headers(response)
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] Exception during file info request: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    # DELETE - удаление файла
    elif request.method == "DELETE":
        logger.info(f"[{request_id}] Received request: DELETE /v1/files/{file_id}")
        try:
            # Если файлы хранятся в memcached, удаляем файл из списка
            deleted = False
            if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
                try:
                    user_key = f"user:{api_key}"
                    user_files_json = safe_memcached_operation('get', user_key)
                    
                    if user_files_json:
                        try:
                            if isinstance(user_files_json, str):
                                user_files = json.loads(user_files_json)
                            elif isinstance(user_files_json, bytes):
                                user_files = json.loads(user_files_json.decode('utf-8'))
                            else:
                                user_files = user_files_json
                                
                            # Фильтруем список, исключая файл с указанным ID
                            new_user_files = [f for f in user_files if f.get("id") != file_id]
                            
                            # Если список изменился, сохраняем обновленный список
                            if len(new_user_files) < len(user_files):
                                safe_memcached_operation('set', user_key, json.dumps(new_user_files))
                                logger.info(f"[{request_id}] Deleted file {file_id} from user's files in memcached")
                                deleted = True
                        except Exception as e:
                            logger.error(f"[{request_id}] Error updating user files in memcached: {str(e)}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")
            
            # Возвращаем ответ об успешном удалении (даже если файл не был найден)
            response_data = {
                "id": file_id,
                "object": "file",
                "deleted": True
            }
            
            response = make_response(jsonify(response_data))
            set_response_headers(response)
            return response
            
        except Exception as e:
            logger.error(f"[{request_id}] Exception during file deletion: {str(e)}")
            return jsonify({"error": str(e)}), 500


@app.route("/v1/files/<file_id>/content", methods=["GET", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file_content(file_id):
    """
    Маршрут для получения содержимого файла
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: GET /v1/files/{file_id}/content")
    
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]
    
    try:
        # В 1min.ai нет API для получения содержимого файла по ID
        # Возвращаем ошибку
        logger.error(f"[{request_id}] File content retrieval not supported")
        return jsonify({"error": "File content retrieval not supported"}), 501
        
    except Exception as e:
        logger.error(f"[{request_id}] Exception during file content request: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Функция-обертка для безопасного доступа к memcached
def safe_memcached_operation(operation, *args, **kwargs):
    """
    Безопасно выполняет операцию с memcached, обрабатывая возможные ошибки.
    
    Args:
        operation: Имя метода memcached для выполнения
        *args, **kwargs: Аргументы для метода
        
    Returns:
        Результат операции или None в случае ошибки
    """
    if 'MEMCACHED_CLIENT' not in globals() or MEMCACHED_CLIENT is None:
        return None
        
    try:
        method = getattr(MEMCACHED_CLIENT, operation, None)
        if method is None:
            logger.error(f"Memcached operation '{operation}' not found")
            return None
            
        return method(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in memcached operation '{operation}': {str(e)}")
        return None


def delete_all_files_task():
    """
    Функция для периодического удаления всех файлов пользователей
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting scheduled files cleanup task")
    
    try:
        # Получаем всех пользователей с файлами из memcached
        if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
            # Получаем все ключи, которые начинаются с "user:"
            try:
                keys = []
                
                # Вместо сканирования slabs используем список известных пользователей
                # который должен сохраняться при загрузке файлов
                known_users = safe_memcached_operation('get', 'known_users_list')
                if known_users:
                    try:
                        if isinstance(known_users, str):
                            user_list = json.loads(known_users)
                        elif isinstance(known_users, bytes):
                            user_list = json.loads(known_users.decode('utf-8'))
                        else:
                            user_list = known_users
                            
                        for user in user_list:
                            user_key = f"user:{user}" if not user.startswith("user:") else user
                            if user_key not in keys:
                                keys.append(user_key)
                    except Exception as e:
                        logger.warning(f"[{request_id}] Failed to parse known users list: {str(e)}")
                
                logger.info(f"[{request_id}] Found {len(keys)} user keys for cleanup")
                
                # Удаляем файлы для каждого пользователя
                for user_key in keys:
                    try:
                        api_key = user_key.replace("user:", "")
                        user_files_json = safe_memcached_operation('get', user_key)
                        
                        if not user_files_json:
                            continue
                            
                        user_files = []
                        try:
                            if isinstance(user_files_json, str):
                                user_files = json.loads(user_files_json)
                            elif isinstance(user_files_json, bytes):
                                user_files = json.loads(user_files_json.decode('utf-8'))
                            else:
                                user_files = user_files_json
                        except:
                            continue
                        
                        logger.info(f"[{request_id}] Cleaning up {len(user_files)} files for user {api_key[:8]}...")
                        
                        # Удаляем каждый файл
                        for file_info in user_files:
                            file_id = file_info.get("id")
                            if file_id:
                                try:
                                    delete_url = f"{ONE_MIN_ASSET_URL}/{file_id}"
                                    headers = {"API-KEY": api_key}
                                    
                                    delete_response = api_request("DELETE", delete_url, headers=headers)
                                    
                                    if delete_response.status_code == 200:
                                        logger.info(f"[{request_id}] Scheduled cleanup: deleted file {file_id}")
                                    else:
                                        logger.error(f"[{request_id}] Scheduled cleanup: failed to delete file {file_id}: {delete_response.status_code}")
                                except Exception as e:
                                    logger.error(f"[{request_id}] Scheduled cleanup: error deleting file {file_id}: {str(e)}")
                        
                        # Очищаем список файлов пользователя
                        safe_memcached_operation('set', user_key, json.dumps([]))
                        logger.info(f"[{request_id}] Cleared files list for user {api_key[:8]}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error processing user {user_key}: {str(e)}")
            except Exception as e:
                logger.error(f"[{request_id}] Error getting keys from memcached: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Error in scheduled cleanup task: {str(e)}")
    
    # Запланировать следующее выполнение через час
    cleanup_timer = threading.Timer(3600, delete_all_files_task)
    cleanup_timer.daemon = True
    cleanup_timer.start()
    logger.info(f"[{request_id}] Scheduled next cleanup in 1 hour")

# Запускаем задачу при старте сервера
if __name__ == "__main__":
    # Запускаем задачу удаления файлов
    delete_all_files_task()
    
    # Запускаем приложение
    internal_ip = socket.gethostbyname(socket.gethostname())
    try:
        response = requests.get("https://api.ipify.org")
        public_ip = response.text
    except:
        public_ip = "Не удалось определить"
        
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

def split_text_for_streaming(text, chunk_size=6):
    """
    Разбивает текст на небольшие части для эмуляции потокового вывода.
    
    Args:
        text (str): Текст для разбиения
        chunk_size (int): Приблизительный размер частей в словах
        
    Returns:
        list: Список частей текста
    """
    if not text:
        return [""]
    
    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Группируем предложения в чанки
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        
        # Если текущий чанк пуст или добавление предложения не превысит лимит слов
        if not current_chunk or current_word_count + words_in_sentence <= chunk_size:
            current_chunk.append(sentence)
            current_word_count += words_in_sentence
        else:
            # Формируем чанк и начинаем новый
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words_in_sentence
    
    # Добавляем последний чанк, если он не пуст
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Если чанков нет (разбиение не сработало), возвращаем весь текст целиком
    if not chunks:
        return [text]
    
    return chunks
