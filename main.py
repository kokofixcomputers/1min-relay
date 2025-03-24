import base64
import datetime
import hashlib
import io
import json
import logging
import os
import re
import socket
import sys
import tempfile
import time
import traceback
import uuid
import warnings
from io import BytesIO

import coloredlogs
import requests
import tiktoken
from flask import Flask, request, jsonify, make_response, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from pymemcache.client.base import Client
import pytz

# Suppress warnings from flask_limiter
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter.extension")

# Set default port and debug mode
PORT = os.environ.get('PORT', 5001)
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() == 'true'

# Create a logger object
logger = logging.getLogger("1min-relay")

# Install coloredlogs with desired log level
coloredlogs.install(level='DEBUG', logger=logger)

warnings.filterwarnings("ignore", category=FutureWarning, module="mistral_common.tokens.tokenizers.mistral")

model = 'gpt-4o-mini'


# noinspection PyBroadException
def check_memcached_connection(host='memcached', port=11211):
    # noinspection PyBroadException
    try:
        client = Client((host, port))
        client.set('test_key', 'test_value')
        if client.get('test_key') == b'test_value':
            client.delete('test_key')  # Clean up
            return True
        else:
            return False
    except:
        return False


logger.info('''
  _ __  __ _      ___     _           
 / |  \/  (_)_ _ | _ \___| |__ _ _  _ 
 | | |\/| | | ' \|   / -_) / _` | || |
 |_|_|  |_|_|_||_|_|_\___|_\__,_|\_, |
                                 |__/ ''')


# noinspection PyShadowingNames
def calculate_token(sentence, model="DEFAULT"):
    """Calculate the number of tokens in a sentence based on the specified model."""

    if model.startswith("mistral"):
        # Initialize the Mistral tokenizer
        MistralTokenizer.v3(is_tekken=True)
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


# Initialize Flask app
app = Flask(__name__)
if check_memcached_connection():
    limiter = Limiter(
        get_remote_address,
        app=app,
        storage_uri="memcached://memcached:11211",  # Connect to Memcached created with docker
    )
else:
    # Used for ratelimiting without memcached
    limiter = Limiter(
        get_remote_address,
        app=app,
    )
    logger.warning("Memcached is not available. Using in-memory storage for rate limiting. Not-Recommended")

# 1minAI API endpoints
ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features?isStreaming=true"
ONE_MIN_ASSET_URL = "https://api.1min.ai/api/assets"

# Define the models that are available for use
ALL_ONE_MIN_AVAILABLE_MODELS = [
    "deepseek-chat",
    "o1-preview",
    "o1-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "claude-instant-1.2",
    "claude-2.1",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-nemo",
    "open-mistral-7b",
    # STT
    # "whisper-1",
    # TTS
    # "alloy",
    # Replicate
    "meta/llama-2-70b-chat",
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3.1-405b-instruct",
    "command"
]

# Define the models that support vision inputs
vision_supported_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo"
]

# Define models that support tool use (function calling)
tools_supported_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

# Define models that support text-to-speech
tts_supported_models = [
    "alloy"
]

stt_supported_models = [
    "whisper-1"
]

# Define models that support web search
web_search_supported_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-vision-preview",
    "gpt-4",
    "gpt-3.5-turbo"
]

# Define models that support image generation
image_generation_models = [
    "dall-e-3",
    "dall-e-2",
    "stable-diffusion-xl-1024-v1-0",
    "midjourney"
]

# Default values
SUBSET_OF_ONE_MIN_PERMITTED_MODELS = ["mistral-nemo", "gpt-4o", "deepseek-chat"]
PERMIT_MODELS_FROM_SUBSET_ONLY = False

# Read environment variables
one_min_models_env = os.getenv("SUBSET_OF_ONE_MIN_PERMITTED_MODELS")  # e.g. "mistral-nemo,gpt-4o,deepseek-chat"
permit_not_in_available_env = os.getenv("PERMIT_MODELS_FROM_SUBSET_ONLY")  # e.g. "True" or "False"

# Parse or fall back to defaults
if one_min_models_env:
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS = one_min_models_env.split(",")

if permit_not_in_available_env and permit_not_in_available_env.lower() == "true":
    PERMIT_MODELS_FROM_SUBSET_ONLY = True

# Combine into a single list
AVAILABLE_MODELS = []
AVAILABLE_MODELS.extend(SUBSET_OF_ONE_MIN_PERMITTED_MODELS)

# Default model to use
DEFAULT_MODEL = "gpt-4o-mini"


# noinspection PyShadowingNames
def map_model_to_openai(model):
    """Map 1minAI model name to OpenAI compatible model name"""
    if model == "mistral-nemo":
        return "mistral-7b-text-chat"
    elif model.startswith("gpt-"):
        return model  # Already in OpenAI format
    elif model.startswith("claude-"):
        return model  # Return as is
    else:
        # For other models, return as is but prefixed with 1min-
        return f"1min-{model}"


# noinspection PyShadowingNames
def error_handler(code, model=None, key=None, detail=None):
    # Handle errors in OpenAI-Structured Error
    error_codes = {  # Internal Error Codes
        1002: {"message": f"The model {model} does not exist.", "type": "invalid_request_error", "param": None,
               "code": "model_not_found", "http_code": 400},
        1020: {"message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.",
               "type": "authentication_error", "param": None, "code": "invalid_api_key", "http_code": 401},
        1021: {"message": "Invalid Authentication", "type": "invalid_request_error", "param": None, "code": None,
               "http_code": 401},
        1212: {"message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.",
               "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1044: {"message": f"This model does not support image inputs.", "type": "invalid_request_error", "param": None,
               "code": "model_not_supported", "http_code": 400},
        1045: {"message": f"This model does not support tool use.", "type": "invalid_request_error", "param": None,
               "code": "model_not_supported", "http_code": 400},
        1046: {"message": f"This model does not support text-to-speech.", "type": "invalid_request_error",
               "param": None, "code": "model_not_supported", "http_code": 400},
        1412: {"message": f"No message provided.", "type": "invalid_request_error", "param": "messages",
               "code": "invalid_request_error", "http_code": 400},
        1423: {"message": f"No content in last message.", "type": "invalid_request_error", "param": "messages",
               "code": "invalid_request_error", "http_code": 400},
        1500: {"message": f"1minAI API error: {detail}", "type": "api_error", "param": None, "code": "api_error",
               "http_code": 500},
        1600: {"message": f"Unsupported feature: {detail}", "type": "invalid_request_error", "param": None,
               "code": "unsupported_feature", "http_code": 400},
        1700: {"message": f"Invalid file format: {detail}", "type": "invalid_request_error", "param": None,
               "code": "invalid_file_format", "http_code": 400},
    }
    error_data = {k: v for k, v in error_codes.get(code, {
        "message": f"Unknown error: {detail}" if detail else "Unknown error", "type": "unknown_error", "param": None,
        "code": None}).items() if k != "http_code"}  # Remove http_code from the error data
    logger.error(f"An error has occurred while processing the user's request. Error code: {code}")
    return jsonify({"error": error_data}), error_codes.get(code, {}).get("http_code",
                                                                         400)  # Return the error data without
    # http_code inside the payload and get the http_code to return.


def handle_options_request():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response, 204


def extract_api_key():
    """Extract API key from Authorization header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    return auth_header.split(" ")[1]


def set_response_headers(response):
    """Set common response headers"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return error_handler(1212)
    if request.method == 'GET':
        internal_ip = socket.gethostbyname(socket.gethostname())
        return "Congratulations! Your API is working! You can now make requests to the API.\n\nEndpoint: " + internal_ip + ':5001/v1'


@app.route('/v1/models')
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
                "created": 1727389042
            }
            for model_name in ALL_ONE_MIN_AVAILABLE_MODELS
        ]
    else:
        one_min_models_data = [
            {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
            for model_name in SUBSET_OF_ONE_MIN_PERMITTED_MODELS
        ]
    models_data.extend(one_min_models_data)

    # Add TTS models
    # tts_models_data = [
    #     {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
    #     for model_name in tts_supported_models
    # ]
    # models_data.extend(tts_models_data)

    # Add STT models
    # stt_models_data = [
    #     {"id": model_name, "object": "model", "owned_by": "1minai", "created": 1727389042}
    #     for model_name in stt_supported_models
    # ]
    # models_data.extend(stt_models_data)

    return jsonify({"data": models_data, "object": "list"})


# noinspection DuplicatedCode
def format_conversation_history(messages, new_input, system_prompt=None):
    """
    Formats the conversation history into a structured string.

    Args:
        messages (list): List of message dictionaries from the request
        new_input (str): The new user input message
        system_prompt (str): Optional system prompt to prepend

    Returns:
        str: Formatted conversation history
    """
    formatted_history = []

    # Add system prompt if provided
    if system_prompt:
        formatted_history.append(f"System: {system_prompt}\n")

    formatted_history.append("Conversation History:\n")

    for message in messages:
        role = message.get('role', '').capitalize()
        content = message.get('content', '')

        # Handle potential list content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if 'text' in item:
                    text_parts.append(item['text'])
                elif 'type' in item and item['type'] == 'text':
                    text_parts.append(item.get('text', ''))
            content = '\n'.join(text_parts)

        formatted_history.append(f"{role}: {content}")

    # Append additional messages only if there are existing messages
    if messages:  # Save credits if it is the first message.
        formatted_history.append(
            "Respond like normal. The conversation history will be automatically updated on the next MESSAGE. DO NOT "
            "ADD User: or Assistant: to your output. Just respond like normal.")
        formatted_history.append("User Message:\n")
    formatted_history.append(new_input)

    return '\n'.join(formatted_history)


# noinspection PyShadowingNames
def extract_images_from_message(messages, api_key, model):
    """
    Extracts images from message content and uploads them to 1minAI

    Returns:
        tuple: (user_input as text, list of image paths, image flag)
    """
    image = False
    image_paths = []
    user_input = ""
    has_ignored_images = False

    if not messages:
        return user_input, image_paths, image

    last_message = messages[-1]
    content = last_message.get('content', '')

    # If content is not a list, return as is
    if not isinstance(content, list):
        return content, image_paths, image

    # Process multi-modal content (text + images)
    text_parts = []

    for item in content:
        # Extract text
        if 'text' in item:
            text_parts.append(item['text'])
        elif 'type' in item and item['type'] == 'text':
            text_parts.append(item.get('text', ''))

        # Extract and process images
        try:
            if 'image_url' in item:
                if model not in vision_supported_models:
                    # If model doesn't support images, ignore them and log warning
                    has_ignored_images = True
                    logger.warning(f"Model {model} does not support images in 1minAI API. Images will be ignored.")
                    continue

                # Process base64 images
                if isinstance(item['image_url'], dict) and 'url' in item['image_url']:
                    image_url = item['image_url']['url']
                    if image_url.startswith("data:image/"):
                        # Handle base64 encoded image
                        mime_type = re.search(r'data:(image/[^;]+);base64,', image_url)
                        mime_type = mime_type.group(1) if mime_type else 'image/png'
                        base64_image = image_url.split(",")[1]
                        binary_data = base64.b64decode(base64_image)

                        # Create a BytesIO object
                        image_data = BytesIO(binary_data)
                    else:
                        # Handle URL images
                        response = requests.get(image_url)
                        response.raise_for_status()
                        image_data = BytesIO(response.content)
                        mime_type = response.headers.get('content-type', 'image/png')

                    # Upload to 1minAI
                    headers = {"API-KEY": api_key}

                    # Add a detail parameter for better image analysis of people
                    if 'detail' in item and item['detail'] == 'high':
                        # High detail for photos with people or complex scenes
                        "high"
                    else:
                        "auto"

                    # Generate a unique filename
                    image_filename = f"relay_image_{uuid.uuid4()}.{mime_type.split('/')[-1]}"

                    files = {
                        'asset': (image_filename, image_data, mime_type)
                    }

                    asset_response = requests.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
                    asset_response.raise_for_status()

                    # Get image path and add to list
                    image_path = asset_response.json()['fileContent']['path']
                    image_paths.append(image_path)
                    image = True

                    # For models that need specific guidance on image analysis
                    if 'claude' in model and not any(text in item.get('text', '') for text in
                                                     ["describe", "what do you see", "analyze", "explain"]):
                        # Add image analysis instructions for Claude models to improve recognition of people
                        analysis_prompts = [
                            "Describe this image in detail. If there are people in the image, describe them. If there "
                            "is text in the image, read it.",
                            "What do you see in this image? Please describe all elements, including any people, "
                            "objects, text, and scenery."
                        ]

                        # Add the prompt to the text part only if no other text was provided
                        if not text_parts:
                            text_parts.append(analysis_prompts[0])
        except Exception as e:
            logger.error(f"Error processing image: {str(e)[:100]}")
            # Continue to process other content even if one image fails

    # Combine all text parts
    user_input = '\n'.join(text_parts)

    # If images were ignored, add a warning to the beginning of the message
    if has_ignored_images:
        user_input = f"Note: This model cannot process images. Please use one of these models instead: {', '.join(vision_supported_models[:3])} and others.\n\n{user_input}"

    return user_input, image_paths, image


def process_tools(tools, tool_choice="auto", model="gpt-4o-mini"):
    """
    Process tools configuration from OpenAI format to 1minAI format

    Args:
        tools: List of tools in OpenAI format
        tool_choice: Tool choice parameter
        model: The model to use

    Returns:
        dict: Tools configuration in 1minAI format
    """
    logger.info(f"Processing tools for model {model}: {len(tools)} tools, tool_choice={tool_choice}")

    # Convert OpenAI tools format to 1minAI format
    one_min_tools = []
    
    # Проверяем наличие и исправляем имена инструментов при необходимости
    for tool in tools:
        # Currently only 'function' type is supported
        if tool.get('type', 'function') == 'function':
            function_def = tool.get('function', tool)  # Handle both formats
            tool_name = function_def.get('name', '')
            
            # Исправляем имена инструментов для совместимости с 1minAI
            if tool_name == 'execute_python' or tool_name == 'execute_python_code':
                logger.info(f"Standardizing tool name '{tool_name}' to 'execute_python_code'")
                tool_name = 'execute_python_code'
            elif tool_name == 'get_current_datetime':
                logger.info("Converting 'get_current_datetime' tool name to 'get_datetime'")
                tool_name = 'get_datetime'
            
            # Проверяем, что имя инструмента входит в список поддерживаемых
            supported_tool_names = ['get_datetime', 'execute_python_code', 'web_search']
            if tool_name not in supported_tool_names:
                logger.warning(f"Tool name '{tool_name}' is not in the list of supported tool names: {supported_tool_names}")
            
            one_min_tool = {
                "name": tool_name,
                "description": function_def.get('description', ''),
                "parameters": function_def.get('parameters', {})
            }
            one_min_tools.append(one_min_tool)
            logger.debug(f"Added tool: {tool_name}")

    # Process tool_choice
    auto_invoke = True  # Default
    if tool_choice == "none":
        auto_invoke = False
        logger.info("Tool auto-invocation is disabled")
    elif isinstance(tool_choice, dict) and tool_choice.get('type') == 'function':
        # Specific function is requested
        function_name = tool_choice.get('function', {}).get('name', '')
        if function_name:
            logger.info(f"Specific tool requested: {function_name}")
            # 1minAI doesn't directly support this, but we can add a note
            pass

    result = {
        "tools": one_min_tools,
        "autoInvoke": auto_invoke
    }
    
    logger.info(f"Processed {len(one_min_tools)} tools, auto_invoke={auto_invoke}")
    return result


def handle_get_datetime(parameters):
    """
    Get the current date and time function
    """
    
    
    logger.info(f"handle_get_datetime called with parameters: {parameters}")
    
    # Всегда используем текущую дату, без кэширования
    current_datetime = datetime.datetime.now(datetime.timezone.utc)
    logger.info(f"Current UTC datetime: {current_datetime}")
    
    # Обрабатываем параметр timezone
    timezone_param = parameters.get("timezone", "UTC")
    logger.info(f"Requested timezone: {timezone_param}")
    
    try:
        # Пытаемся определить часовой пояс
        if timezone_param != "UTC":
            # Проверяем формат названия города
            if "/" not in timezone_param:
                # Преобразуем обычные названия городов в названия часовых поясов
                city_to_timezone = {
                    "москва": "Europe/Moscow",
                    "нью-йорк": "America/New_York",
                    "токио": "Asia/Tokyo",
                    "лондон": "Europe/London",
                    "париж": "Europe/Paris",
                    "берлин": "Europe/Berlin",
                    "пекин": "Asia/Shanghai",
                    "сидней": "Australia/Sydney"
                }
                
                # Приводим к нижнему регистру для корректного сравнения
                timezone_lower = timezone_param.lower()
                if timezone_lower in city_to_timezone:
                    timezone_param = city_to_timezone[timezone_lower]
                    logger.info(f"Converted city name to timezone: {timezone_param}")
                else:
                    # Если не удалось найти соответствие, используем UTC
                    logger.warning(f"Unknown city: {timezone_param}, using UTC")
                    timezone_param = "UTC"
            
            # Применяем часовой пояс
            try:
                tz = pytz.timezone(timezone_param)
                current_datetime = current_datetime.replace(tzinfo=datetime.timezone.utc).astimezone(tz)
                logger.info(f"Current datetime after timezone conversion: {current_datetime}")
            except Exception as e:
                logger.error(f"Error converting timezone: {str(e)}")
                # В случае ошибки используем UTC
                timezone_param = "UTC"
    except Exception as tz_error:
        logger.error(f"Error processing timezone: {str(tz_error)}")
        timezone_param = "UTC"
    
    # Явно форматируем дату для лога
    formatted_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Formatted date: {formatted_date}")
    
    # Словарь месяцев на русском языке
    month_names_ru = {
        1: "января", 2: "февраля", 3: "марта", 4: "апреля", 
        5: "мая", 6: "июня", 7: "июля", 8: "августа", 
        9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
    }
    
    # Расширенный словарь с более детальной информацией
    result = {
        "datetime": formatted_date,
        "timezone": timezone_param,
        "year": current_datetime.year,
        "month": current_datetime.month,
        "month_name": month_names_ru.get(current_datetime.month, current_datetime.strftime("%B")),
        "day": current_datetime.day,
        "hour": current_datetime.hour,
        "minute": current_datetime.minute,
        "second": current_datetime.second,
        "weekday": current_datetime.strftime("%A"),
        "iso_date": current_datetime.date().isoformat()
    }
    
    logger.info(f"Returning datetime result: {result}")
    return result


def execute_python_code(code, timeout=10):
    """
    Безопасно выполняет Python код с ограничением по времени

    Args:
        code (str): Python код для выполнения
        timeout (int): Максимальное время выполнения в секундах

    Returns:
        dict: Результат выполнения кода
    """
    import subprocess
    import tempfile
    import os
    from concurrent.futures import ThreadPoolExecutor
    
    # Создаем временный файл для кода
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as temp_file:
        temp_file_path = temp_file.name
        # Добавляем код в файл с перехватом вывода
        wrapper_code = f"""
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Ограничиваем импорты для безопасности
__import__ = __builtins__.__import__
restricted_modules = ['os', 'subprocess', 'sys', 'importlib', 'shutil', 'ctypes', 'socket']
def restricted_import(name, *args, **kwargs):
    if name in restricted_modules:
        raise ImportError(f"Импорт модуля {{name}} запрещен по соображениям безопасности")
    return __import__(name, *args, **kwargs)
__builtins__.__import__ = restricted_import

# Перехватываем стандартный вывод и ошибки
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()

# Выполняем пользовательский код
user_result = None
with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
    try:
{chr(10).join(['        ' + line for line in code.split(chr(10))])}
        
        # Для совместимости с предыдущими версиями
        if 'result' in locals():
            user_result = result
    except Exception as e:
        print(f"Ошибка: {{str(e)}}", file=sys.stderr)

# Выводим результаты в JSON формате для легкой обработки
import json
print("\\n__CODE_EXECUTION_RESULT__\\n")
print(json.dumps({{"stdout": stdout_buffer.getvalue(), 
                  "stderr": stderr_buffer.getvalue(),
                  "result": str(user_result) if user_result is not None else None}}))
"""
        temp_file.write(wrapper_code)
    
    try:
        # Выполняем код в отдельном процессе
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                subprocess.run,
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Ждем результат с таймаутом
            process_result = future.result(timeout=timeout + 1)  # +1 для запаса
            
            # Обрабатываем результат
            output = process_result.stdout
            error = process_result.stderr
            
            # Извлекаем JSON результат
            if "__CODE_EXECUTION_RESULT__" in output:
                result_json = output.split("__CODE_EXECUTION_RESULT__")[1].strip()
                try:
                    execution_result = json.loads(result_json)
                    return execution_result
                except json.JSONDecodeError:
                    return {
                        "stdout": output,
                        "stderr": error,
                        "error": "Ошибка парсинга JSON результата"
                    }
            else:
                return {
                    "stdout": output,
                    "stderr": error,
                    "error": "Не удалось получить результат выполнения кода"
                }
                
    except subprocess.TimeoutExpired:
        return {"error": f"Превышен лимит времени выполнения ({timeout} сек)"}
    except Exception as e:
        return {"error": f"Ошибка выполнения кода: {str(e)}"}
    finally:
        # Удаляем временный файл
        try:
            os.remove(temp_file_path)
        except:
            pass


def handle_execute_python_code(params):
    """
    Handle the execute_python_code tool call

    Args:
        params (dict): Tool parameters

    Returns:
        dict: Tool response
    """
    try:
        logger.info(f"handle_execute_python_code called with parameters: {params}")
        
        code = params.get('code', '')
        timeout = int(params.get('timeout', 10))

        # Limit timeout to reasonable values
        if timeout < 1:
            timeout = 1
        elif timeout > 30:
            timeout = 30

        if not code:
            logger.warning("No code provided in execute_python_code request")
            return {"error": "No code provided"}

        # Execute the code
        logger.info(f"Executing Python code with timeout {timeout} seconds")
        logger.debug(f"Python code to execute: {code[:200]}..." if len(code) > 200 else f"Python code to execute: {code}")
        result = execute_python_code(code, timeout)
        logger.info(f"Python code execution result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error handling execute_python_code tool: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": f"Error executing Python code: {str(e)}"}


def handle_web_search(params):
    """
    Handle web search function call
    
    Args:
        params (dict): Function parameters
            - query (str): Search query
            - num_results (int, optional): Number of results to return
            - search_type (str, optional): Type of search (web, news, etc.)
            
    Returns:
        dict: Search results
    """
    try:
        # Validate parameters
        if 'query' not in params:
            return {"error": "Missing required parameter: query"}
        
        query = params.get('query', '')
        num_results = min(params.get('num_results', 3), 10)  # Limit to max 10 results
        search_type = params.get('search_type', 'web')
        
        logger.info(f"Performing web search for query: {query}")
        
        # Simulate web search results (in production, you would call a real search API)
        # This is a placeholder for demonstration purposes
        search_results = {
            "query": query,
            "results": [],
            "search_time": datetime.datetime.now().isoformat()
        }
        
        # Call a real search API here if available
        try:
            # Setup a basic web search using a search engine API
            # For example purposes, we'll just return some mock data
            for i in range(num_results):
                search_results["results"].append({
                    "title": f"Search Result {i+1} for '{query}'",
                    "snippet": f"This is a snippet of content related to {query}...",
                    "url": f"https://example.com/result-{i+1}"
                })
        except Exception as search_e:
            logger.error(f"Error performing web search: {str(search_e)}")
            return {"error": str(search_e)}
            
        logger.info(f"Web search completed with {len(search_results['results'])} results")
        return search_results
    except Exception as e:
        logger.error(f"Error in web search handler: {str(e)}")
        return {"error": str(e)}


# noinspection PyShadowingNames
def process_tts_request(request_data):
    """
    Process text-to-speech request

    Args:
        request_data (dict): Request data

    Returns:
        Response: Flask response with audio data or error
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return error_handler(1021)

    api_key = auth_header.split(" ")[1]

    # Extract parameters
    input_text = request_data.get('input', '')
    model = request_data.get('model', 'nova')
    voice = request_data.get('voice', 'alloy')
    response_format = request_data.get('response_format', 'mp3')
    speed = request_data.get('speed', 1.0)

    if not input_text:
        return error_handler(1412, detail="No input text provided for TTS")

    if model not in tts_supported_models:
        return error_handler(1046, model=model)

    # Prepare request to 1minAI API using unified endpoint
    headers = {
        "API-KEY": api_key,
        "Content-Type": "application/json"
    }

    # Использование единого эндпоинта с типом TEXT_TO_SPEECH
    payload = {
        "type": "TEXT_TO_SPEECH",
        "model": model,
        "promptObject": {
            "text": input_text,
            "voice": voice,
            "speed": speed,
            "response_format": response_format
        }
    }

    try:
        logger.info(f"Sending TTS request to: {ONE_MIN_API_URL}")
        logger.debug(f"TTS payload: {json.dumps(payload)}")

        response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
        response.raise_for_status()

        # Get the audio data
        tts_response = response.json()
        audio_url = tts_response.get('audioUrl')

        if not audio_url:
            return error_handler(1500, detail="No audio URL returned from 1minAI TTS API")

        # Download the audio file
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()

        # Create response
        flask_response = make_response(audio_response.content)
        flask_response.headers['Content-Type'] = f'audio/{response_format}'

        return flask_response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in TTS processing: {str(e)}")
        return error_handler(1500, detail=str(e))


def process_stt_request():
    """
    Process speech-to-text request

    Returns:
        Response: Flask response with transcription or error
    """
    global model
    logger.info("Processing speech-to-text request")

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Missing or invalid authorization header")
        return error_handler(1021)

    api_key = auth_header.split(" ")[1]

    # Check if file is uploaded
    if 'file' not in request.files:
        logger.error("No audio file in request")
        return error_handler(1700, detail="No audio file provided")

    if model not in stt_supported_models:
        return error_handler(1046, model=model)

    audio_file = request.files['file']

    # Get the originally requested model (which will process the transcribed text)
    original_model = request.form.get('model', 'gpt-4o-mini')
    logger.info(f"Original model requested: {original_model}")

    # Для STT всегда используем whisper-1
    model = "whisper-1"
    logger.info(f"Processing audio file: {audio_file.filename} with {model} for transcription")

    # Save the file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_file_path = temp_file.name
        logger.debug(f"Saved audio file temporarily at: {temp_file_path}")

    try:
        # Upload to 1minAI
        headers = {
            "API-KEY": api_key,
        }
        files = {
            'asset': (audio_file.filename, open(temp_file_path, 'rb'), audio_file.content_type)
        }

        logger.info(f"Uploading audio file to 1minAI asset URL: {ONE_MIN_ASSET_URL}")
        asset_response = requests.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
        asset_response.raise_for_status()

        # Get audio path
        asset_data = asset_response.json()
        logger.debug(f"Asset response: {json.dumps(asset_data)[:200]}...")
        audio_path = asset_data['fileContent']['path']
        logger.info(f"Audio file uploaded successfully. Path: {audio_path}")

        # Отправка запроса на транскрипцию в 1minAI
        features_url = "https://api.1min.ai/api/features"
        payload = {
            "type": "SPEECH_TO_TEXT",
            "model": model,
            "promptObject": {
                "audioUrl": audio_path,
                "response_format": "text"
            }
        }

        logger.info(f"Sending transcription request to: {features_url}")
        logger.debug(f"Transcription payload: {payload}")

        # Используем тот же формат заголовка API-KEY
        headers = {'API-KEY': api_key}
        transcription_response = requests.post(
            features_url,
            json=payload,
            headers=headers
        )

        # Проверка статуса ответа
        logger.debug(f"Speech-to-text response status: {transcription_response.status_code}")
        logger.debug(f"Speech-to-text response headers: {transcription_response.headers}")
        logger.debug(f"Speech-to-text response body: {transcription_response.text}")

        if transcription_response.status_code != 200:
            logger.error(f"Error from 1minAI API: {transcription_response.text}")
            return jsonify({"error": f"Error from 1minAI API: {transcription_response.text}"}), 500

        # Обработка ответа от 1minAI
        ""
        response_data = transcription_response.json()

        # Извлечение текста из ответа от 1minAI (формат может различаться)
        if 'aiRecord' in response_data and 'aiRecordDetail' in response_data['aiRecord']:
            details = response_data['aiRecord']['aiRecordDetail']
            if 'resultObject' in details and isinstance(details['resultObject'], list) and details['resultObject']:
                transcript = "".join(details['resultObject'])
                logger.info(f"Transcription successful: {transcript}")
            else:
                logger.error("Invalid response format from 1minAI API")
                return jsonify({"error": "Invalid response format from 1minAI API"}), 500
        else:
            logger.error("Invalid response format from 1minAI API")
            return jsonify({"error": "Invalid response format from 1minAI API"}), 500

        # Если требуется также получить ответ на транскрибированный текст
        response_model = request.form.get('response_model', None)
        if response_model:
            logger.info(f"Forwarding transcribed text to original model: {response_model}")

            # Отправляем запрос к модели для генерации ответа
            try:
                # Устанавливаем API URL для запроса
                api_url = "https://api.1min.ai/api/features"

                # Формируем данные запроса
                payload = {
                    "type": "CHAT_WITH_AI",
                    "model": response_model,
                    "promptObject": {
                        "prompt": transcript,
                        "streaming": False
                    }
                }

                logger.debug(f"Sending request to {api_url} with payload: {payload}")

                # Отправляем запрос с правильным заголовком API-KEY
                headers = {'API-KEY': api_key}
                response = requests.post(
                    api_url,
                    json=payload,
                    headers=headers
                )

                # Обрабатываем ответ
                if response.status_code == 200:
                    response_data = response.json()
                    logger.debug(f"Received response: {response_data}")

                    # Извлекаем содержимое из ответа
                    content = ""
                    if 'aiRecord' in response_data and 'aiRecordDetail' in response_data['aiRecord']:
                        details = response_data['aiRecord']['aiRecordDetail']
                        if 'resultObject' in details:
                            result_obj = details['resultObject']
                            if isinstance(result_obj, dict) and 'content' in result_obj:
                                content = result_obj['content']
                            elif isinstance(result_obj, str):
                                content = result_obj
                            elif isinstance(result_obj, list) and len(result_obj) > 0:
                                content = "".join([str(item) for item in result_obj])

                    logger.info(f"Extracted content from response: {content[:100]}...")

                    # Подготавливаем ответ в формате OpenAI API
                    openai_format_response = {
                        "id": f"chatcmpl-{str(uuid.uuid4())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": response_model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": content
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": len(transcript.split()),
                            "completion_tokens": len(content.split()),
                            "total_tokens": len(transcript.split()) + len(content.split())
                        }
                    }

                    logger.debug(f"Successfully processed audio response with {len(content.split())} tokens")
                    logger.debug(f"Returning OpenAI format response: {openai_format_response}")

                    # Здесь сохраняем полный ответ для отладки
                    with open('/tmp/last_voice_response.json', 'w', encoding='utf-8') as f:
                        json.dump(openai_format_response, f, ensure_ascii=False, indent=2)

                    # Удаление временного файла
                    try:
                        os.remove(temp_file_path)
                        logger.debug(f"Removed temporary file: {temp_file_path}")
                    except Exception as e:
                        logger.error(f"Error removing temporary file: {str(e)}")

                    return jsonify(openai_format_response)
                else:
                    logger.error(f"Error response from AI API: {response.status_code} - {response.text}")
                    return jsonify({
                        "error": f"Error response from AI API: {response.status_code}"
                    }), 500

            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return jsonify({
                    "error": f"Error generating response: {str(e)}"
                }), 500

        # Формирование ответа в формате OpenAI API
        openai_format_response = {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": transcript
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(transcript.split()),
                "total_tokens": len(transcript.split())
            }
        }

        # Удаление временного файла
        try:
            os.remove(temp_file_path)
            logger.debug(f"Removed temporary file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")

        return jsonify(openai_format_response)

    except Exception as e:
        logger.error(f"Error processing audio transcription: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error processing audio transcription: {str(e)}"}), 500


@app.route('/v1/audio/transcriptions', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def audio_transcriptions():
    """Endpoint для транскрипции аудио в текст с использованием модели Whisper-1.
    Также поддерживает перенаправление текста на выбранную модель для генерации ответа.
    """
    if request.method == 'OPTIONS':
        return handle_options_request()

    return process_stt_request()


@app.route('/v1/audio/speech', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def audio_speech():
    if request.method == 'OPTIONS':
        return handle_options_request()
    request_data = request.json
    return process_tts_request(request_data)
    # Temporarily return an error that the function is disabled
    # return error_handler(1600, detail="TTS functionality is temporarily disabled")


# noinspection PyShadowingNames,DuplicatedCode
@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def conversation():
    if request.method == 'OPTIONS':
        return handle_options_request()

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return error_handler(1021)

    api_key = auth_header.split(" ")[1]
    request_data = request.json
    
    # Проверка запроса на получение даты - прямая обработка без обращения к API
    tools = request_data.get('tools', [])
    print(f"TOOLS: {tools}")
    has_datetime_tool = False
    for tool in tools:
        print(f"Processing tool: {tool}")
        if tool.get('type') == 'function' and tool.get('function', {}).get('name') == 'get_datetime':
            has_datetime_tool = True
            print("FOUND get_datetime TOOL!")
            break
    
    # Если запрашивают дату, сразу возвращаем текущую дату, минуя API
    messages = request_data.get('messages', [])
    print(f"MESSAGES: {messages}")
    has_date_keyword = any("дат" in msg.get('content', '').lower() for msg in messages)
    print(f"Has date keyword: {has_date_keyword}")
    
    if has_datetime_tool and has_date_keyword:
        print("DETECTED DATE REQUEST - RETURNING DIRECT RESPONSE")
        # Получаем текущую дату
        now = datetime.datetime.now()
        print(f"CURRENT DATE: {now}")
        # Словарь месяцев на русском языке
        month_names_ru = {
            1: "января", 2: "февраля", 3: "марта", 4: "апреля", 
            5: "мая", 6: "июня", 7: "июля", 8: "августа", 
            9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
        }
        # Форматируем дату на русском языке
        formatted_date = f"{now.day} {month_names_ru[now.month]} {now.year}"
        date_response = f"Сегодня {formatted_date} года."
        print(f"FORMATTED DATE: {formatted_date}")
        print(f"FULL RESPONSE: {date_response}")
        
        # Создаем прямой ответ
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        direct_response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get('model', DEFAULT_MODEL),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": date_response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,  # примерно
                "completion_tokens": 20,  # примерно
                "total_tokens": 120  # примерно
            }
        }
        
        print(f"DIRECT RESPONSE: {direct_response}")
        # Возвращаем ответ напрямую
        return jsonify(direct_response)
    
    print("Not a date request, proceeding with regular processing")

    headers = {
        "API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if request_data.get('stream', False) else "application/json"
    }

    # Get model
    model = request_data.get('model', 'gpt-4o-mini')

    # Check to see if the TTS model is a model
    if model in tts_supported_models:
        return error_handler(1600, detail="TTS models can only be used with the /v1/audio/speech endpoint")

    # Check to see if the STT model is a model
    if model in stt_supported_models:
        return error_handler(1600, detail="STT models can only be used with the /v1/audio/transcriptions endpoint")

    if PERMIT_MODELS_FROM_SUBSET_ONLY and model not in AVAILABLE_MODELS:
        return error_handler(1002, model)

    # Get messages
    if not messages:
        return error_handler(1412)

    # Add system info to last user message if it was a converted DOC/DOCX file
    last_message = messages[-1]
    if last_message.get('role') == 'user' and 'doc_file_conversion' in request.headers:
        original_filename = request.headers.get('doc_file_conversion')
        # Only modify content if it's a string
        if isinstance(last_message.get('content'), str):
            logger.debug(f"Adding note about DOC/DOCX conversion for file: {original_filename}")
            last_message[
                'content'] += f"\n\n(Примечание: Этот текст был извлечен из файла {original_filename}. Некоторое " \
                            f"форматирование могло быть потеряно при конвертации.)"

    # Extract system message if present
    system_prompt = None
    for msg in messages:
        if msg.get('role') == 'system':
            system_prompt = msg.get('content', '')
            break
    
    # Extract user input from the last message
    try:
        user_input, image_paths, has_image = extract_images_from_message(messages, api_key, model)
    except Exception as e:
        logger.error(f"Error extracting images: {str(e)}")
        # If an error occurs during image processing, continue without images
        user_input = messages[-1].get('content', '')
        if isinstance(user_input, list):
            # Merge the text parts of the message
            text_parts = []
            for item in user_input:
                if 'text' in item:
                    text_parts.append(item['text'])
                elif 'type' in item and item['type'] == 'text':
                    text_parts.append(item.get('text', ''))
            user_input = '\n'.join(text_parts)

        image_paths = []
        has_image = False

        if not user_input:
            return error_handler(1423)

    # Format conversation history
    all_messages = format_conversation_history(messages, user_input, system_prompt)
    prompt_token = calculate_token(str(all_messages))

    # Process tools (function calling)
    tools_config = None
    try:
        # Определяем базовый набор инструментов
        base_tools = request_data.get('tools', [])

        # Для поддерживаемых моделей автоматически добавляем стандартные инструменты
        if model in tools_supported_models:
            # Добавляем стандартные инструменты, если они не указаны явно
            standard_tools = [
                {"type": "function", "function": {"name": "get_datetime", "description": "Get current date and time"}},
                {"type": "function", "function": {"name": "execute_python_code", "description": "Execute Python code"}},
                {"type": "function", "function": {"name": "web_search", "description": "Search the web"}}
            ]

            # Если инструменты не заданы, используем стандартные
            if not base_tools:
                base_tools = standard_tools
            # Иначе добавляем стандартные, если их еще нет
            else:
                existing_tool_names = [t.get('function', {}).get('name', '') for t in base_tools if
                                       t.get('type') == 'function']
                for std_tool in standard_tools:
                    if std_tool['function']['name'] not in existing_tool_names:
                        base_tools.append(std_tool)

            tools_config = process_tools(
                base_tools,
                request_data.get('tool_choice', 'auto'),
                model
            )
            logger.info(
                f"Added tools configuration for supported model {model}: {[t.get('function', {}).get('name', '') for t in base_tools if t.get('type') == 'function']}")
        else:
            # Для неподдерживаемых моделей пробуем обработать только явно указанные инструменты
            if base_tools:
                try:
                    tools_config = process_tools(
                        base_tools,
                        request_data.get('tool_choice', 'auto'),
                        model
                    )
                    logger.warning(f"Processing tools for potentially unsupported model {model}")
                except Exception as tool_e:
                    logger.warning(f"Failed to process tools for model {model}: {str(tool_e)}")
                    # Продолжаем без инструментов
                    pass
    except Exception as e:
        logger.error(f"Error processing tools: {str(e)}")
        # Продолжаем без инструментов при ошибке
        pass

    # Check for web search
    use_web_search = request_data.get('web_search', False)
    num_of_site = request_data.get('num_of_site', 1)
    max_word = request_data.get('max_word', 500)
    
    # Автоматически включаем веб-поиск для поддерживаемых моделей, если в запросе есть ключевые слова
    if model in web_search_supported_models and not use_web_search:
        # Проверяем наличие ключевых слов для поиска в интернете
        search_keywords = ['найди', 'поищи', 'search', 'найти', 'поиск', 'погугли', 'загугли', 'ищи', 'интернете',
                           'internet', 'интернета', 'искать', 'интернет', 'интернетом', 'google', 'browse', 'find',
                           'узнай', 'почитай', 'прочитай', 'уточни', 'проверь', 'check', 'онлайн', 'online', 'confirm']
        content = messages[-1].get('content', '')

        # Проверка типа контента и обработка соответствующим образом
        if isinstance(content, list):
            # Для мультимодального контента извлекаем текст
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
            user_message = ' '.join(text_parts).lower()
        else:
            # Для обычного текстового контента
            user_message = content.lower()

        if any(keyword in user_message for keyword in search_keywords):
            use_web_search = True
            logger.info(f"Automatically enabling web search for model {model} based on user message")

    if use_web_search and model not in web_search_supported_models:
        logger.warning(f"Model {model} does not support web search, ignoring web search parameter")
        use_web_search = False

    # Prepare base payload
    payload = {
        "model": model,
        "promptObject": {
            "prompt": all_messages,
            "isMixed": False,
            "webSearch": use_web_search
        }
    }
    
    # Добавляем дополнительные параметры для webSearch
    if use_web_search:
        payload["promptObject"]["numOfSite"] = num_of_site
        payload["promptObject"]["maxWord"] = max_word

    # Set request type based on content
    if has_image:
        payload["type"] = "CHAT_WITH_IMAGE"
        payload["promptObject"]["imageList"] = image_paths
    else:
        payload["type"] = "CHAT_WITH_AI"

    # Add tools if configured
    if tools_config:
        payload["toolsConfig"] = tools_config

    # Add additional parameters
    if 'temperature' in request_data:
        payload["temperature"] = request_data['temperature']

    if 'top_p' in request_data:
        payload["topP"] = request_data['top_p']

    if 'max_tokens' in request_data:
        payload["maxTokens"] = request_data['max_tokens']

    # Add conversationId if provided
    if 'conversation_id' in request_data:
        payload["conversationId"] = request_data['conversation_id']

    # Add metadata if provided
    if 'metadata' in request_data:
        payload["metadata"] = request_data['metadata']

    # Handle response format
    response_format = request_data.get('response_format', {})
    if response_format.get('type') == 'json_object':
        payload["responseFormat"] = "json"

    logger.debug(f"Processing {prompt_token} prompt tokens with model {model}")

    # Check for cases where we need to disable streaming
    stream_enabled = request_data.get('stream', False)

    # Disable streaming for claude-instant-1.2 only, since it doesn't work with it for sure
    if model == "claude-instant-1.2" and stream_enabled:
        logger.warning(f"Model {model} might have issues with streaming, falling back to non-streaming mode")
        stream_enabled = False

    # For all other models we will try streaming if requested

    if not stream_enabled:
        # Non-Streaming Response
        logger.debug("Non-Streaming AI Response")
        try:
            response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            one_min_response = response.json()

            transformed_response = transform_response(one_min_response, request_data, prompt_token)
            flask_response = make_response(jsonify(transformed_response))
            set_response_headers(flask_response)

            return flask_response, 200
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return error_handler(1020, key="[REDACTED]")
            return error_handler(1500, detail=str(e))

    else:
        # Streaming Response
        logger.debug("Streaming AI Response")
        try:
            response_stream = requests.post(ONE_MIN_CONVERSATION_API_STREAMING_URL,
                                            data=json.dumps(payload),
                                            headers=headers,
                                            stream=True)
            response_stream.raise_for_status()

            return Response(
                stream_response(response_stream, request_data, prompt_token),
                mimetype='text/event-stream')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return error_handler(1020, key="[REDACTED]")

            # If you get an error while streaming, try without streaming
            logger.warning(f"Streaming request failed for model {model}, trying non-streaming mode")
            try:
                response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
                response.raise_for_status()
                one_min_response = response.json()

                transformed_response = transform_response(one_min_response, request_data, prompt_token)
                flask_response = make_response(jsonify(transformed_response))
                set_response_headers(flask_response)

                return flask_response, 200
            except requests.exceptions.HTTPError as retry_e:
                if retry_e.response.status_code == 401:
                    return error_handler(1020, key="[REDACTED]")
                return error_handler(1500, detail=str(retry_e))

        except Exception as e:
            return error_handler(1500, detail=str(e))

def transform_response(one_min_response, request_data, prompt_token):
    """
    Преобразует ответ от 1minAI API в формат OpenAI API

    Args:
        one_min_response (dict): Ответ от 1minAI API
        request_data (dict): Данные запроса
        prompt_token (int): Количество токенов в запросе

    Returns:
        dict: Ответ в формате OpenAI API
    """
    try:
        logger.debug(f"Transforming 1minAI response to OpenAI format")
        logger.debug(f"Response data: {json.dumps(one_min_response)[:1000]}")
        
        # Получаем модель из запроса или используем значение по умолчанию
        model = request_data.get('model', DEFAULT_MODEL)
        
        # Создаем базовый ответ
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": []
        }

        # Получаем содержимое ответа от 1minAI
        content = ""
        tool_calls = None
        
        try:
            # Извлекаем результат из формата ответа 1minAI
            if 'aiRecord' in one_min_response and 'aiRecordDetail' in one_min_response['aiRecord']:
                details = one_min_response['aiRecord']['aiRecordDetail']
                logger.debug(f"Found aiRecordDetail: {json.dumps(details)[:1000]}")
                
                # Обработка вызовов инструментов
                if 'toolResults' in details and details['toolResults']:
                    tool_results = details['toolResults']
                    logger.debug(f"Found tool results: {json.dumps(tool_results)[:1000]}")
                    
                    # Если есть вызовы инструмента get_datetime
                    for tool_result in tool_results:
                        if tool_result.get('name') == 'get_datetime':
                            # Форматируем дату на русском языке
                            try:
                                result = tool_result.get('result', {})
                                day = result.get('day', '')
                                month_name = result.get('month_name', '')
                                year = result.get('year', '')
                                formatted_date = f"Сегодня {day} {month_name} {year} года."
                                logger.info(f"Setting formatted date in response: {formatted_date}")
                                content = formatted_date
                            except Exception as e:
                                logger.error(f"Error formatting date: {str(e)}")
                                content = str(result)
                        elif tool_result.get('name') == 'execute_python_code':
                            # Обрабатываем результат выполнения Python кода
                            try:
                                result = tool_result.get('result', {})
                                logger.info(f"Setting Python execution result in response: {result}")
                                stdout = result.get('stdout', '')
                                stderr = result.get('stderr', '')
                                error = result.get('error', '')
                                
                                if error:
                                    content = f"Ошибка выполнения кода: {error}\n\nСтандартный вывод:\n{stdout}\n\nСтандартный вывод ошибок:\n{stderr}"
                                else:
                                    content = f"Результат выполнения кода:\n\n{stdout}"
                                    if stderr:
                                        content += f"\n\nСтандартный вывод ошибок:\n{stderr}"
                            except Exception as e:
                                logger.error(f"Error processing Python execution result: {str(e)}")
                                content = f"Ошибка обработки результата выполнения кода: {str(e)}"
                
                    # Преобразуем вызовы инструментов в формат OpenAI
                    if not content and 'resultObject' in details:
                        content = details['resultObject'].get('content', '')
                
                # Если содержимое не получено из вызовов инструментов, получаем его из результата
                if not content and 'resultObject' in details:
                    result_obj = details['resultObject']
                    logger.debug(f"Processing resultObject: {json.dumps(result_obj)[:1000]}")
                    
                    if isinstance(result_obj, dict) and 'content' in result_obj:
                        content = result_obj['content']
                    elif isinstance(result_obj, str):
                        content = result_obj
                    elif isinstance(result_obj, list) and len(result_obj) > 0:
                        content = "".join([str(item) for item in result_obj])
            
            elif isinstance(one_min_response, dict) and 'detail' in one_min_response:
                # Обработка ошибок от API
                logger.error(f"API returned error: {one_min_response['detail']}")
                content = f"Ошибка API: {one_min_response['detail']}"
            else:
                # Неизвестный формат ответа
                logger.warning(f"Unknown response format: {json.dumps(one_min_response)[:1000]}")
                content = "Извините, получен неизвестный формат ответа от API."
                
        except Exception as e:
            logger.error(f"Error extracting content from 1minAI response: {str(e)}")
            logger.error(traceback.format_exc())
            content = f"Извините, произошла ошибка при обработке ответа: {str(e)}"
        
        # Добавляем выбор в ответ
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }
        
        # Если есть вызовы инструментов, добавляем их
        if tool_calls:
            choice["message"]["tool_calls"] = tool_calls
        
        openai_response["choices"].append(choice)
        
        # Подсчитываем количество токенов в ответе
        try:
            completion_token = calculate_token(content)
            total_token = prompt_token + completion_token
            
            openai_response["usage"] = {
                "prompt_tokens": prompt_token,
                "completion_tokens": completion_token,
                "total_tokens": total_token
            }
        except Exception as e:
            logger.error(f"Error calculating tokens: {str(e)}")
            openai_response["usage"] = {
                "prompt_tokens": prompt_token,
                "completion_tokens": len(content.split()),
                "total_tokens": prompt_token + len(content.split())
            }
        
        return openai_response
    
    except Exception as e:
        # Добавляем подробное логирование для отладки
        logger.error(f"CRITICAL ERROR in transform_response: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error(f"Request data: {json.dumps(request_data)[:1000]}")
        logger.error(f"One min response: {json.dumps(one_min_response)[:5000]}")
        
        # Возвращаем базовый ответ с информацией об ошибке
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get('model', DEFAULT_MODEL),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Произошла критическая ошибка: {str(e)}"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token,
                "completion_tokens": 20,
                "total_tokens": prompt_token + 20
            }
        }


def stream_response(response_stream, request_data, prompt_token):
    """
    Обрабатывает потоковый ответ от 1minAI API

    Args:
        response_stream: Потоковый ответ от 1minAI API
        request_data (dict): Данные запроса
        prompt_token (int): Количество токенов в запросе

    Yields:
        str: Строки в формате SSE для OpenAI streaming
    """
    model = request_data.get('model', DEFAULT_MODEL)
    content = ""
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    total_content = ""
    
    try:
        for chunk in response_stream.iter_lines():
            if chunk and chunk.strip():
                # Пропускаем префикс data:
                if chunk.startswith(b'data:'):
                    chunk = chunk[5:].strip()
                
                # Пропускаем пустые чанки или "[DONE]"
                if not chunk or chunk == b'[DONE]':
                    continue
                
                try:
                    chunk_data = json.loads(chunk)
                    
                    # Извлекаем содержимое из формата 1minAI
                    if 'responseData' in chunk_data and 'content' in chunk_data['responseData']:
                        content = chunk_data['responseData']['content']
                        
                        # Формируем чанк в формате OpenAI
                        delta = {"role": "assistant"}
                        if content:
                            delta["content"] = content
                            total_content += content
                        
                        response_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None
                                }
                            ]
                        }
                        
                        # Отправляем чанк
                        yield f"data: {json.dumps(response_chunk)}\n\n"
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse chunk as JSON: {chunk}")
                    continue
        
        # Отправляем финальный чанк с finish_reason
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        
        # Добавляем использование токенов
        try:
            completion_token = calculate_token(total_content)
            total_token = prompt_token + completion_token
            
            final_chunk["usage"] = {
                "prompt_tokens": prompt_token,
                "completion_tokens": completion_token,
                "total_tokens": total_token
            }
        except Exception as e:
            logger.error(f"Error calculating tokens: {str(e)}")
            final_chunk["usage"] = {
                "prompt_tokens": prompt_token,
                "completion_tokens": len(total_content.split()),
                "total_tokens": prompt_token + len(total_content.split())
            }
        
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"Ошибка: {str(e)}"},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

# The main function to start the server
if __name__ == "__main__":
    # Set the logging level
    if os.environ.get('DEBUG') == 'True':
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Enable Flask debugging
    app.debug = True
    DEBUG_MODE = True

    # Start the server
    print("Starting 1minAI Relay server on port", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
    print("Server shutdown.")  # This line should not be reached unless the server is stopped
