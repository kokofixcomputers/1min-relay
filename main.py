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

import coloredlogs
import printedcolors
import requests
import tiktoken
from flask import Flask, request, jsonify, make_response, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from pymemcache.client.base import Client
from waitress import serve

# Suppress warnings from flask_limiter
warnings.filterwarnings(
    "ignore", category=UserWarning, module="flask_limiter.extension"
)

# Create a logger object
logger = logging.getLogger("1min-relay")

# Install coloredlogs with desired log level
coloredlogs.install(level="DEBUG", logger=logger)


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
else:
    # Used for ratelimiting without memcached
    limiter = Limiter(
        get_remote_address,
        app=app,
    )


ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features/stream"
ONE_MIN_ASSET_URL = "https://api.1min.ai/api/assets"

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
    "whisper-1",  # Распознавание речи
    # "tts-1",     # Синтез речи
    # "tts-1-hd",  # Синтез речи HD
    #
    "dall-e-2",  # Генерация изображений
    "dall-e-3",  # Генерация изображений
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
    # "google-tts",  # Синтез речи
    # "latest_long",  # Распознавание речи
    # "latest_short",  # Распознавание речи
    # "phone_call",  # Распознавание речи
    # "telephony",  # Распознавание речи
    # "telephony_short",  # Распознавание речи
    # "medical_dictation",  # Распознавание речи
    # "medical_conversation",  # Распознавание речи
    "chat-bison@002",
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
    # "stable-image",  # StabilityAI - Генерация изображений
    "stable-diffusion-xl-1024-v1-0",  # StabilityAI - Генерация изображений
    "stable-diffusion-v1-6",  # StabilityAI - Генерация изображений
    # "esrgan-v1-x2plus",  # StabilityAI - Улучшение изображений
    # "stable-video-diffusion",  # StabilityAI - Генерация видео
    "phoenix",  # Leonardo.ai - 6b645e3a-d64f-4341-a6d8-7a3690fbf042
    "lightning-xl",  # Leonardo.ai - b24e16ff-06e3-43eb-8d33-4416c2d75876
    "anime-xl",  # Leonardo.ai - e71a1c2f-4f80-4800-934f-2c68979d8cc8
    "diffusion-xl",  # Leonardo.ai - 1e60896f-3c26-4296-8ecc-53e2afecc132
    "kino-xl",  # Leonardo.ai - aa77f04e-3eec-4034-9c07-d0f619684628
    "vision-xl",  # Leonardo.ai - 5c232a9e-9061-4777-980a-ddc8e65647c6
    "albedo-base-xl",  # Leonardo.ai - 2067ae52-33fd-4a82-bb92-c2c55e7d2786
    # "clipdrop",  # Clipdrop.co - Обработка изображений
    "midjourney",  # Midjourney - Генерация изображений
    "midjourney_6_1",  # Midjourney - Генерация изображений
    # "methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5",  # Replicate - Image to Prompt
    # "cjwbw/damo-text-to-video:1e205ea73084bd17a0a3b43396e49ba0d6bc2e754e9283b2df49fad2dcf95755",  # Replicate - Text to Video
    # "lucataco/animate-diff:beecf59c4aee8d81bf04f0381033dfa10dc16e845b4ae00d281e2fa377e48a9f",  # Replicate - Animation
    # "lucataco/hotshot-xl:78b3a6257e16e4b241245d65c8b2b81ea2e1ff7ed4c55306b511509ddbfd327a",  # Replicate - Video
    "flux-schnell",  # Replicate - Flux "black-forest-labs/flux-schnell"
    "flux-dev",  # Replicate - Flux Dev "black-forest-labs/flux-dev"
    "flux-pro",  # Replicate - Flux Pro "black-forest-labs/flux-pro"
    "flux-1.1-pro",  # Replicate - Flux Pro 1.1 "black-forest-labs/flux-1.1-pro"
    # "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",  # Replicate - Music Generation
    # "luma",  # TTAPI - Luma
    # "Qubico/image-toolkit",  # TTAPI - Image Toolkit
    # "suno",  # TTAPI - Suno Music
    # "kling",  # TTAPI - Kling
    # "music-u",  # TTAPI - Music U
    # "music-s",  # TTAPI - Music S
    # "elevenlabs-tts"  # ElevenLabs - TTS
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

# Default values
SUBSET_OF_ONE_MIN_PERMITTED_MODELS = ["mistral-nemo", "gpt-4o", "deepseek-chat"]
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
            payload = {
                "type": "CHAT_WITH_IMAGE",
                "model": model,
                "promptObject": {
                    "prompt": all_messages,
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
    logger.info(f"[{request_id}] Creating conversation with files: {file_ids}")

    try:
        payload = {
            "title": title,
            "type": "CHAT_WITH_PDF",
            "model": model,
            "fileList": file_ids,
        }

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}

        response = api_request(
            "POST", ONE_MIN_CONVERSATION_API_URL, json=payload, headers=headers
        )

        if response.status_code != 200:
            logger.error(
                f"[{request_id}] Failed to create conversation: {response.status_code} - {response.text}"
            )
            return None

        response_data = response.json()

        # Ищем ID беседы в разных местах ответа
        conversation_id = None
        if "conversation" in response_data and "uuid" in response_data["conversation"]:
            conversation_id = response_data["conversation"]["uuid"]
        elif "id" in response_data:
            conversation_id = response_data["id"]
        elif "uuid" in response_data:
            conversation_id = response_data["uuid"]

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

        # Журналируем начало запроса
        logger.debug(f"[{request_id}] Processing chat completion request")

        # Проверяем, содержит ли запрос изображения
        image = False
        image_paths = []
        messages = request_data.get("messages", [])

        if not messages:
            logger.error(f"[{request_id}] No messages provided in request")
            return ERROR_HANDLER(1412)

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

        # Если есть file_ids, используем CHAT_WITH_PDF
        if file_ids and len(file_ids) > 0:
            logger.debug(
                f"[{request_id}] Creating CHAT_WITH_PDF request with {len(file_ids)} files"
            )

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
            payload = {"conversationId": conversation_id, "message": all_messages}

            # Используем URL для бесед вместо общего API URL
            api_url = f"{ONE_MIN_CONVERSATION_API_URL}/{conversation_id}/message"
            headers = {"API-KEY": api_key, "Content-Type": "application/json"}

            # Выполнение запроса в зависимости от stream
            if not request_data.get("stream", False):
                try:
                    response = api_request(
                        "POST", api_url, json=payload, headers=headers
                    )
                    logger.debug(
                        f"[{request_id}] Response status code: {response.status_code}"
                    )

                    if response.status_code != 200:
                        if response.status_code == 401:
                            return ERROR_HANDLER(1020, key=api_key)
                        try:
                            error_content = response.json()
                            logger.error(
                                f"[{request_id}] Error response: {error_content}"
                            )
                        except:
                            logger.error(
                                f"[{request_id}] Could not parse error response as JSON"
                            )
                        return ERROR_HANDLER(response.status_code)

                    one_min_response = response.json()
                    # Извлекаем ответ из структуры сообщения беседы
                    if "message" in one_min_response:
                        message_content = one_min_response["message"].get("content", "")
                        one_min_response = {"resultObject": [message_content]}

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
                # Потоковый запрос для беседы с документами
                try:
                    response = api_request(
                        "POST", api_url, json=payload, headers=headers
                    )

                    if response.status_code != 200:
                        if response.status_code == 401:
                            return ERROR_HANDLER(1020, key=api_key)
                        return ERROR_HANDLER(response.status_code)

                    one_min_response = response.json()
                    # Извлекаем текст ответа из структуры сообщения беседы
                    if "message" in one_min_response:
                        message_content = one_min_response["message"].get("content", "")
                    else:
                        message_content = "Не удалось извлечь содержание ответа из беседы с документами"

                    # Возвращаем эмулированный поток
                    return Response(
                        emulate_stream_response(
                            message_content, request_data, model, prompt_token
                        ),
                        content_type="text/event-stream",
                    )
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Exception during streaming request emulation: {str(e)}"
                    )
                    return jsonify({"error": str(e)}), 500

            # Прерываем обработку, так как ответ уже отправлен
            return

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


@app.route("/v1/images/generations", methods=["POST", "OPTIONS"])
@limiter.limit("500 per minute")
def generate_image():
    if request.method == "OPTIONS":
        return handle_options_request()

    # Создаем уникальный ID для запроса
    request_id = str(uuid.uuid4())
    logger.debug(f"[{request_id}] Processing image generation request")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]
    headers = {"API-KEY": api_key, "Content-Type": "application/json"}

    request_data = request.json
    model = request_data.get("model", "dall-e-2").strip()
    logger.debug(f"[{request_id}] Using model: {model}")

    # Преобразование параметров OpenAI в формат 1min.ai
    prompt = request_data.get("prompt", "")

    if model == "dall-e-3":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dall-e-3",
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                "size": request_data.get("size", "1024x1024"),
                "quality": request_data.get("quality", "hd"),
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
                "size": request_data.get("size", "1024x1024"),
            },
        }
    elif model == "stable-diffusion-xl-1024-v1-0":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "stable-diffusion-xl-1024-v1-0",
            "promptObject": {
                "prompt": prompt,
                "samples": request_data.get("n", 1),
                "size": request_data.get("size", "1024x1024"),
                "cfg_scale": request_data.get("cfg_scale", 7),
                "clip_guidance_preset": request_data.get(
                    "clip_guidance_preset", "NONE"
                ),
                "seed": request_data.get("seed", 0),
                "steps": request_data.get("steps", 30),
            },
        }
    elif model == "midjourney":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "midjourney",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "aspect_ratio": request_data.get("size", "1:1"),
            },
        }
    elif model == "midjourney_6_1" or model == "midjourney-6.1":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "midjourney_6_1",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "aspect_ratio": request_data.get("size", "1:1"),
                "style": request_data.get("style", None),
                "quality": request_data.get("quality", None),
                "stylize": request_data.get("stylize", None),
                "weird": request_data.get("weird", None),
                "chaos": request_data.get("chaos", None),
                "isNiji6": request_data.get("isNiji6", False),
            },
        }
        # Удаляем None параметры
        payload["promptObject"] = {
            k: v for k, v in payload["promptObject"].items() if v is not None
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
    elif model in ["black-forest-labs/flux-schnell", "flux-schnell"]:
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "black-forest-labs/flux-schnell",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "aspect_ratio": request_data.get("aspect_ratio", "1:1"),
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
                "aspect_ratio": request_data.get("aspect_ratio", "1:1"),
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
                "aspect_ratio": request_data.get("aspect_ratio", "1:1"),
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
                "aspect_ratio": request_data.get("aspect_ratio", "1:1"),
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
                "num_outputs": request_data.get("n", 1),
                "width": request_data.get("width", 1024),
                "height": request_data.get("height", 1024),
                "negative_prompt": request_data.get("negative_prompt", ""),
            },
        }
        # Удаляем пустые параметры
        if not payload["promptObject"]["negative_prompt"]:
            del payload["promptObject"]["negative_prompt"]
    elif model in [
        "b24e16ff-06e3-43eb-8d33-4416c2d75876",
        "lightning-xl",
    ]:  # Leonardo.ai - Lightning XL
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "b24e16ff-06e3-43eb-8d33-4416c2d75876",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "width": request_data.get("width", 1024),
                "height": request_data.get("height", 1024),
                "negative_prompt": request_data.get("negative_prompt", ""),
            },
        }
        # Удаляем пустые параметры
        if not payload["promptObject"]["negative_prompt"]:
            del payload["promptObject"]["negative_prompt"]
    elif model in [
        "5c232a9e-9061-4777-980a-ddc8e65647c6",
        "vision-xl",
    ]:  # Leonardo.ai - Vision XL
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "5c232a9e-9061-4777-980a-ddc8e65647c6",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "width": request_data.get("width", 1024),
                "height": request_data.get("height", 1024),
                "negative_prompt": request_data.get("negative_prompt", ""),
            },
        }
        # Удаляем пустые параметры
        if not payload["promptObject"]["negative_prompt"]:
            del payload["promptObject"]["negative_prompt"]
    elif model in [
        "e71a1c2f-4f80-4800-934f-2c68979d8cc8",
        "anime-xl",
    ]:  # Leonardo.ai - Anime XL
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "e71a1c2f-4f80-4800-934f-2c68979d8cc8",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "width": request_data.get("width", 1024),
                "height": request_data.get("height", 1024),
                "negative_prompt": request_data.get("negative_prompt", ""),
            },
        }
        # Удаляем пустые параметры
        if not payload["promptObject"]["negative_prompt"]:
            del payload["promptObject"]["negative_prompt"]
    elif model in [
        "1e60896f-3c26-4296-8ecc-53e2afecc132",
        "diffusion-xl",
    ]:  # Leonardo.ai - Diffusion XL
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "1e60896f-3c26-4296-8ecc-53e2afecc132",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "width": request_data.get("width", 1024),
                "height": request_data.get("height", 1024),
                "negative_prompt": request_data.get("negative_prompt", ""),
            },
        }
        # Удаляем пустые параметры
        if not payload["promptObject"]["negative_prompt"]:
            del payload["promptObject"]["negative_prompt"]
    elif model in [
        "aa77f04e-3eec-4034-9c07-d0f619684628",
        "kino-xl",
    ]:  # Leonardo.ai - Kino XL
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "aa77f04e-3eec-4034-9c07-d0f619684628",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "width": request_data.get("width", 1024),
                "height": request_data.get("height", 1024),
                "negative_prompt": request_data.get("negative_prompt", ""),
            },
        }
        # Удаляем пустые параметры
        if not payload["promptObject"]["negative_prompt"]:
            del payload["promptObject"]["negative_prompt"]
    elif model in [
        "2067ae52-33fd-4a82-bb92-c2c55e7d2786",
        "albedo-base-xl",
    ]:  # Leonardo.ai - Albedo Base XL
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "2067ae52-33fd-4a82-bb92-c2c55e7d2786",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "width": request_data.get("width", 512),
                "height": request_data.get("height", 512),
                "negative_prompt": request_data.get("negative_prompt", ""),
            },
        }
        # Удаляем пустые параметры
        if not payload["promptObject"]["negative_prompt"]:
            del payload["promptObject"]["negative_prompt"]

    else:
        logger.error(f"[{request_id}] Invalid model: {model}")
        return ERROR_HANDLER(1002, model)

    try:
        logger.debug(
            f"[{request_id}] Sending image generation request to {ONE_MIN_API_URL}"
        )
        logger.debug(f"[{request_id}] Payload: {json.dumps(payload)[:200]}...")

        response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
        logger.debug(
            f"[{request_id}] Image generation response status code: {response.status_code}"
        )

        if response.status_code != 200:
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            return (
                jsonify({"error": response.json().get("error", "Unknown error")}),
                response.status_code,
            )

        one_min_response = response.json()

        # Преобразование ответа 1min.ai в формат OpenAI
        try:
            image_url = (
                one_min_response.get("aiRecord", {})
                .get("aiRecordDetail", {})
                .get("resultObject", [""])[0]
            )

            if not image_url:
                # Попробуем другие пути извлечения URL
                if "resultObject" in one_min_response:
                    image_url = (
                        one_min_response["resultObject"][0]
                        if isinstance(one_min_response["resultObject"], list)
                        else one_min_response["resultObject"]
                    )

            if not image_url:
                logger.error(
                    f"[{request_id}] Could not extract image URL from API response"
                )
                return (
                    jsonify({"error": "Could not extract image URL from API response"}),
                    500,
                )

            logger.debug(
                f"[{request_id}] Successfully generated image: {image_url[:50]}..."
            )

            openai_response = {
                "created": int(time.time()),
                "data": [{"url": image_url}],
            }

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
                return (
                    jsonify(
                        {
                            "error": asset_response.json().get(
                                "error", "Failed to upload image"
                            )
                        }
                    ),
                    asset_response.status_code,
                )

            image_path = asset_response.json()["fileContent"]["path"]
            logger.debug(f"[{request_id}] Successfully uploaded image: {image_path}")
        finally:
            session.close()

        # Создание вариации в зависимости от модели
        if model == "dall-e-2":
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "dall-e-2",
                "promptObject": {"imageUrl": image_path, "n": int(n), "size": size},
            }
        elif model == "dall-e-3":
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "dall-e-3",
                "promptObject": {
                    "imageUrl": image_path,
                    "n": int(n),
                    "size": size,
                    "style": request.form.get("style", "vivid"),
                    "quality": request.form.get("quality", "hd"),
                },
            }
        elif model == "midjourney":
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "midjourney",
                "promptObject": {
                    "imageUrl": image_path,
                    "mode": request.form.get("mode", "fast"),
                    "n": int(n),
                    "isNiji6": request.form.get("isNiji6", "false").lower() == "true",
                    "aspect_width": int(request.form.get("aspect_width", 1)),
                    "aspect_height": int(request.form.get("aspect_height", 1)),
                    "maintainModeration": request.form.get(
                        "maintainModeration", "true"
                    ).lower()
                    == "true",
                },
            }
        elif model == "midjourney_6_1" or model == "midjourney-6.1":
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "midjourney_6_1",
                "promptObject": {
                    "imageUrl": image_path,
                    "mode": request.form.get("mode", "fast"),
                    "n": int(n),
                    "isNiji6": request.form.get("isNiji6", "false").lower() == "true",
                    "aspect_width": int(request.form.get("aspect_width", 1)),
                    "aspect_height": int(request.form.get("aspect_height", 1)),
                    "maintainModeration": request.form.get(
                        "maintainModeration", "true"
                    ).lower()
                    == "true",
                    "custom_zoom": request.form.get("custom_zoom"),
                    "style": request.form.get("style"),
                },
            }
            # Удаляем None параметры
            payload["promptObject"] = {
                k: v for k, v in payload["promptObject"].items() if v is not None
            }
        elif model in [
            "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
            "phoenix",
        ]:  # Leonardo.ai - Phoenix
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
                "promptObject": {
                    "imageUrl": image_path,
                    "n": int(n),
                    "width": int(request.form.get("width", 1024)),
                    "height": int(request.form.get("height", 1024)),
                    "negative_prompt": request.form.get("negative_prompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negative_prompt"]:
                del payload["promptObject"]["negative_prompt"]
        elif model in [
            "b24e16ff-06e3-43eb-8d33-4416c2d75876",
            "lightning-xl",
        ]:  # Leonardo.ai - Lightning XL
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "b24e16ff-06e3-43eb-8d33-4416c2d75876",
                "promptObject": {
                    "imageUrl": image_path,
                    "n": int(n),
                    "width": int(request.form.get("width", 1024)),
                    "height": int(request.form.get("height", 1024)),
                    "negative_prompt": request.form.get("negative_prompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negative_prompt"]:
                del payload["promptObject"]["negative_prompt"]
        elif model in [
            "e71a1c2f-4f80-4800-934f-2c68979d8cc8",
            "anime-xl",
        ]:  # Leonardo.ai - Anime XL
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "e71a1c2f-4f80-4800-934f-2c68979d8cc8",
                "promptObject": {
                    "imageUrl": image_path,
                    "n": int(n),
                    "width": int(request.form.get("width", 1024)),
                    "height": int(request.form.get("height", 1024)),
                    "negative_prompt": request.form.get("negative_prompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negative_prompt"]:
                del payload["promptObject"]["negative_prompt"]
        elif model in [
            "1e60896f-3c26-4296-8ecc-53e2afecc132",
            "diffusion-xl",
        ]:  # Leonardo.ai - Diffusion XL
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "1e60896f-3c26-4296-8ecc-53e2afecc132",
                "promptObject": {
                    "imageUrl": image_path,
                    "n": int(n),
                    "width": int(request.form.get("width", 1024)),
                    "height": int(request.form.get("height", 1024)),
                    "negative_prompt": request.form.get("negative_prompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negative_prompt"]:
                del payload["promptObject"]["negative_prompt"]
        elif model in [
            "aa77f04e-3eec-4034-9c07-d0f619684628",
            "kino-xl",
        ]:  # Leonardo.ai - Kino XL
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "aa77f04e-3eec-4034-9c07-d0f619684628",
                "promptObject": {
                    "imageUrl": image_path,
                    "n": int(n),
                    "width": int(request.form.get("width", 1024)),
                    "height": int(request.form.get("height", 1024)),
                    "negative_prompt": request.form.get("negative_prompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negative_prompt"]:
                del payload["promptObject"]["negative_prompt"]
        elif model in [
            "5c232a9e-9061-4777-980a-ddc8e65647c6",
            "vision-xl",
        ]:  # Leonardo.ai - Vision XL
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "5c232a9e-9061-4777-980a-ddc8e65647c6",
                "promptObject": {
                    "imageUrl": image_path,
                    "n": int(n),
                    "width": int(request.form.get("width", 1024)),
                    "height": int(request.form.get("height", 1024)),
                    "negative_prompt": request.form.get("negative_prompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negative_prompt"]:
                del payload["promptObject"]["negative_prompt"]
        elif model in [
            "2067ae52-33fd-4a82-bb92-c2c55e7d2786",
            "albedo-base-xl",
        ]:  # Leonardo.ai - Albedo Base XL
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "2067ae52-33fd-4a82-bb92-c2c55e7d2786",
                "promptObject": {
                    "imageUrl": image_path,
                    "n": int(n),
                    "width": int(request.form.get("width", 1024)),
                    "height": int(request.form.get("height", 1024)),
                    "negative_prompt": request.form.get("negative_prompt", ""),
                },
            }
            # Удаляем пустые параметры
            if not payload["promptObject"]["negative_prompt"]:
                del payload["promptObject"]["negative_prompt"]
        else:
            logger.error(f"[{request_id}] Invalid model for variations: {model}")
            return ERROR_HANDLER(1002, model)

        headers["Content-Type"] = "application/json"
        logger.debug(
            f"[{request_id}] Sending image variation request with payload: {json.dumps(payload)[:200]}..."
        )

        response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
        logger.debug(
            f"[{request_id}] Image variation response status code: {response.status_code}"
        )

        if response.status_code != 200:
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            logger.error(
                f"[{request_id}] Error in variation response: {response.text[:200]}"
            )
            return (
                jsonify({"error": response.json().get("error", "Unknown error")}),
                response.status_code,
            )

        one_min_response = response.json()

        # Преобразование ответа 1min.ai в формат OpenAI
        try:
            # Безопасное извлечение URL изображения
            image_url = (
                one_min_response.get("aiRecord", {})
                .get("aiRecordDetail", {})
                .get("resultObject", [""])[0]
            )

            if not image_url:
                # Попробуем другие пути извлечения URL
                if "resultObject" in one_min_response:
                    image_url = (
                        one_min_response["resultObject"][0]
                        if isinstance(one_min_response["resultObject"], list)
                        else one_min_response["resultObject"]
                    )

            if not image_url:
                logger.error(
                    f"[{request_id}] Could not extract variation image URL from API response"
                )
                return (
                    jsonify({"error": "Could not extract image URL from API response"}),
                    500,
                )

            logger.debug(
                f"[{request_id}] Successfully generated image variation: {image_url[:50]}..."
            )

            openai_response = {
                "created": int(time.time()),
                "data": [{"url": image_url}],
            }

            response = make_response(jsonify(openai_response))
            set_response_headers(response)
            return response, 200
        except Exception as e:
            logger.error(
                f"[{request_id}] Error processing image variation response: {str(e)}"
            )
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(
            f"[{request_id}] Exception during image variation request: {str(e)}"
        )
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
    response.headers["Access -Control-Allow-Origin"] = "*"
    response.headers["X-Request-ID"] = str(uuid.uuid4())


def stream_response(response, request_data, model, prompt_tokens, session=None):
    all_chunks = ""

    try:
        for line in response.iter_lines():
            if not line:
                continue

            # Трассировка для отладки
            logger.debug(f"Received line from stream: {line[:100]}")

            try:
                if line.startswith(b"data: "):
                    data_str = line[6:].decode("utf-8")
                    if data_str == "[DONE]":
                        break

                    # Проверяем, является ли это JSON-объектом
                    try:
                        json_data = json.loads(data_str)
                        # Если это полный ответ 1min.ai
                        if "aiRecord" in json_data:
                            result_text = (
                                json_data.get("aiRecord", {})
                                .get("aiRecordDetail", {})
                                .get("resultObject", [""])[0]
                            )
                            chunk_text = result_text
                        else:
                            # Если это просто фрагмент текста в JSON
                            chunk_text = json_data.get("text", data_str)
                    except json.JSONDecodeError:
                        # Если это не JSON, используем как есть
                        chunk_text = data_str
                else:
                    # Для обычного текста
                    try:
                        # Проверяем, не является ли это JSON-объектом
                        raw_text = line.decode("utf-8")
                        try:
                            json_data = json.loads(raw_text)
                            # Если это полный ответ 1min.ai
                            if "aiRecord" in json_data:
                                result_text = (
                                    json_data.get("aiRecord", {})
                                    .get("aiRecordDetail", {})
                                    .get("resultObject", [""])[0]
                                )
                                chunk_text = result_text
                            else:
                                # Если это просто фрагмент текста в JSON
                                chunk_text = json_data.get("text", raw_text)
                        except json.JSONDecodeError:
                            # Если это не JSON, используем как есть
                            chunk_text = raw_text
                    except Exception as e:
                        logger.error(f"Error decoding raw stream chunk: {str(e)}")
                        continue

                # Добавляем текущий фрагмент к общему тексту
                all_chunks += chunk_text

                # Создаем чанк для OpenAI API
                return_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None,
                        }
                    ],
                }

                yield f"data: {json.dumps(return_chunk)}\n\n"

            except Exception as e:
                logger.error(f"Error processing stream chunk: {str(e)}")
                continue

        # Подсчитываем токены после завершения
        tokens = calculate_token(all_chunks)
        logger.debug(
            f"Finished processing streaming response. Completion tokens: {str(tokens)}"
        )
        logger.debug(f"Total tokens: {str(tokens + prompt_tokens)}")

        # Финальный чанк, обозначающий конец потока
        final_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": tokens + prompt_tokens,
            },
        }

        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    finally:
        # Закрываем сессию, если она была передана
        if session:
            try:
                session.close()
                logger.debug("Streaming session closed properly")
            except Exception as e:
                logger.error(f"Error closing streaming session: {str(e)}")


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
    Загружает документ на сервер 1min.ai

    Args:
        file_data: Данные файла (bytes)
        file_name: Имя файла
        api_key: API ключ
        request_id: ID запроса для логирования

    Returns:
        str: ID файла или None в случае ошибки
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Uploading document: {file_name}")

    # Создаем новую сессию для этого запроса
    session = create_session()

    try:
        # Определяем MIME-тип файла на основе расширения
        extension = os.path.splitext(file_name)[1].lower()

        # Словарь MIME-типов для разных расширений файлов
        mime_types = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".csv": "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
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

        # Формируем ответ в формате OpenAI API
        response_data = {
            "id": file_id,
            "object": "file",
            "bytes": len(file_data),
            "created_at": int(time.time()),
            "filename": file_name,
            "purpose": "assistants",
        }

        return jsonify(response_data), 200
    except Exception as e:
        logger.error(f"[{request_id}] Error processing file upload: {str(e)}")
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
def api_request(method, url, **kwargs):
    """
    Выполняет HTTP-запрос с новой сессией и автоматическим закрытием соединения

    Args:
        method: HTTP-метод (GET, POST и т.д.)
        url: URL для запроса
        **kwargs: Дополнительные параметры для requests

    Returns:
        requests.Response: Ответ от сервера
    """
    session = create_session()
    try:
        response = session.request(method, url, **kwargs)
        return response
    finally:
        session.close()


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

            if not result_text:
                logger.error(
                    f"[{request_id}] Could not extract transcription text from API response"
                )
                return jsonify({"error": "Could not extract transcription text"}), 500

            # Формируем ответ в формате OpenAI API
            openai_response = {"text": result_text}

            # Для JSON формата добавляем дополнительные поля
            if response_format == "json":
                openai_response = {
                    "task": "transcribe",
                    "language": language or "en",
                    "duration": 0,  # Не имеем информации о длительности
                    "text": result_text,
                }

            response = make_response(
                jsonify(openai_response) if response_format == "json" else result_text
            )
            set_response_headers(response)
            return response, 200
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

            # Формируем ответ в формате OpenAI API
            openai_response = {"text": result_text}

            # Для JSON формата добавляем дополнительные поля
            if response_format == "json":
                openai_response = {
                    "task": "translate",
                    "language": "en",  # Whisper обычно переводит на английский
                    "duration": 0,  # Не имеем информации о длительности
                    "text": result_text,
                }

            response = make_response(
                jsonify(openai_response) if response_format == "json" else result_text
            )
            set_response_headers(response)
            return response, 200
        except Exception as e:
            logger.error(
                f"[{request_id}] Error processing translation response: {str(e)}"
            )
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Exception during translation request: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    internal_ip = socket.gethostbyname(socket.gethostname())
    response = requests.get("https://api.ipify.org")
    public_ip = response.text
    logger.info(
        f"""{printedcolors.Color.fg.lightcyan}  
Server is ready to serve at:
Internal IP: {internal_ip}:5001
Public IP: {public_ip} (only if you've setup port forwarding on your router.)
Enter this url to OpenAI clients supporting custom endpoint:
{internal_ip}:5001/v1
If does not work, try:
{internal_ip}:5001/v1/chat/completions
{printedcolors.Color.reset}"""
    )
    serve(
        app, host="0.0.0.0", port=5001, threads=6
    )  # Thread has a default of 4 if not specified. We use 6 to increase performance and allow multiple requests at once.
