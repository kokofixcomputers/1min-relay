from flask import Flask, request, jsonify, make_response, Response
import requests
import time
import uuid
import warnings
from waitress import serve
import json
import tiktoken
import socket
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from pymemcache.client.base import Client
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import logging
from io import BytesIO
import coloredlogs
import printedcolors
import base64
import hashlib
from PIL import Image
import random
import string
import traceback
import collections

# Suppress warnings from flask_limiter
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter.extension")

# Create a logger object
logger = logging.getLogger("1min-relay")

# Install coloredlogs with desired log level
coloredlogs.install(level='DEBUG', logger=logger)

def check_memcached_connection():
    """
    Проверяет доступность memcached, сначала в Docker, затем локально
    
    Returns:
        tuple: (bool, str) - (доступен ли memcached, строка подключения или None)
    """
    # Проверяем Docker memcached
    try:
        client = Client(('memcached', 11211))
        client.set('test_key', 'test_value')
        if client.get('test_key') == b'test_value':
            client.delete('test_key')  # Clean up
            logger.info("Using memcached in Docker container")
            return True, "memcached://memcached:11211"
    except Exception as e:
        logger.debug(f"Docker memcached not available: {str(e)}")
    
    # Проверяем локальный memcached
    try:
        client = Client(('127.0.0.1', 11211))
        client.set('test_key', 'test_value')
        if client.get('test_key') == b'test_value':
            client.delete('test_key')  # Clean up
            logger.info("Using local memcached at 127.0.0.1:11211")
            return True, "memcached://127.0.0.1:11211"
    except Exception as e:
        logger.debug(f"Local memcached not available: {str(e)}")
    
    # Если memcached недоступен
    logger.warning("Memcached is not available. Using in-memory storage for rate limiting. Not-Recommended")
    return False, None

logger.info('''
  _ __  __ _      ___     _           
 / |  \/  (_)_ _ | _ \___| |__ _ _  _ 
 | | |\/| | | ' \|   / -_) / _` | || |
 |_|_|  |_|_|_||_|_|_\___|_\__,_|\_, |
                                 |__/ ''')


def calculate_token(sentence, model="DEFAULT"):
    """Calculate the number of tokens in a sentence based on the specified model."""
    
    if model.startswith("mistral"):
        # Initialize the Mistral tokenizer
        tokenizer = MistralTokenizer.v3(is_tekken=True)
        model_name = "open-mistral-nemo" # Default to Mistral Nemo
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

# Добавим кэш для отслеживания обработанных изображений
# Для каждого запроса храним уникальный идентификатор изображения и его путь
IMAGE_CACHE = {}
# Ограничим размер кэша
MAX_CACHE_SIZE = 100

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return ERROR_HANDLER(1212)
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
    return jsonify({"data": models_data, "object": "list"})

def ERROR_HANDLER(code, model=None, key=None):
    # Handle errors in OpenAI-Structued Error
    error_codes = { # Internal Error Codes
        1002: {"message": f"The model {model} does not exist.", "type": "invalid_request_error", "param": None, "code": "model_not_found", "http_code": 400},
        1020: {"message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.", "type": "authentication_error", "param": None, "code": "invalid_api_key", "http_code": 401},
        1021: {"message": "Invalid Authentication", "type": "invalid_request_error", "param": None, "code": None, "http_code": 401},
        1212: {"message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1044: {"message": f"This model does not support image inputs.", "type": "invalid_request_error", "param": None, "code": "model_not_supported", "http_code": 400},
        1412: {"message": f"No message provided.", "type": "invalid_request_error", "param": "messages", "code": "invalid_request_error", "http_code": 400},
        1423: {"message": f"No content in last message.", "type": "invalid_request_error", "param": "messages", "code": "invalid_request_error", "http_code": 400},
    }
    error_data = {k: v for k, v in error_codes.get(code, {"message": "Unknown error", "type": "unknown_error", "param": None, "code": None}).items() if k != "http_code"} # Remove http_code from the error data
    logger.error(f"An error has occurred while processing the user's request. Error code: {code}")
    return jsonify({"error": error_data}), error_codes.get(code, {}).get("http_code", 400) # Return the error data without http_code inside the payload and get the http_code to return.

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
        role = message.get('role', '')
        content = message.get('content', '')
        
        # Handle potential list content
        if isinstance(content, list):
            processed_content = []
            for item in content:
                if 'text' in item:
                    processed_content.append(item['text'])
            content = '\n'.join(processed_content)
        
        if role == 'system':
            formatted_history.append(f"System: {content}")
        elif role == 'user':
            formatted_history.append(f"User: {content}")
        elif role == 'assistant':
            formatted_history.append(f"Assistant: {content}")
    
    # Добавляем новый ввод, если он есть
    if new_input:
        formatted_history.append(f"User: {new_input}")
    
    # Возвращаем только историю диалога без дополнительных инструкций
    return '\n'.join(formatted_history)


@app.route('/v1/chat/completions', methods=['POST'])
@limiter.limit("60 per minute")
def conversation():
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/chat/completions")
    
    if not request.json:
        return jsonify({"error": "Invalid request format"}), 400
    
    # Извлекаем информацию из запроса
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not api_key:
        logger.error(f"[{request_id}] No API key provided")
        return jsonify({"error": "API key required"}), 401
    
    try:
        # Строим payload для запроса
        request_data = request.json.copy()
        
        # Получаем и нормализуем модель
        model = request_data.get('model', '').strip()
        logger.info(f"[{request_id}] Using model: {model}")
        
        # Журналируем начало запроса
        logger.debug(f"[{request_id}] Processing chat completion request")
        
        # Проверяем, содержит ли запрос изображения
        image = False
        image_paths = []
        messages = request_data.get('messages', [])
        
        if not messages:
            logger.error(f"[{request_id}] No messages provided in request")
            return ERROR_HANDLER(1412)
        
        user_input = messages[-1].get('content')
        if not user_input:
            logger.error(f"[{request_id}] No content in last message")
            return ERROR_HANDLER(1423)
        
        # Формируем историю диалога
        all_messages = format_conversation_history(request_data.get('messages', []), request_data.get('new_input', ''))
        
        # Проверка на наличие изображений в последнем сообщении
        if isinstance(user_input, list):
            logger.debug(f"[{request_id}] Processing message with multiple content items (text/images)")
            combined_text = ""
            for i, item in enumerate(user_input):
                if 'text' in item:
                    combined_text += item['text'] + "\n"
                    logger.debug(f"[{request_id}] Added text content from item {i+1}")
                
                if 'image_url' in item:
                    if model not in vision_supported_models + ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']:
                        logger.error(f"[{request_id}] Model {model} does not support images")
                        return ERROR_HANDLER(1044, model)
                    
                    # Создаем хеш URL изображения для кэширования
                    image_key = None
                    image_url = None
                    
                    # Извлекаем URL изображения
                    if isinstance(item['image_url'], dict) and 'url' in item['image_url']:
                        image_url = item['image_url']['url']
                    else:
                        image_url = item['image_url']
                    
                    # Хешируем URL для кэша
                    if image_url:
                        image_key = hashlib.md5(image_url.encode('utf-8')).hexdigest()
                    
                    # Проверяем кэш
                    if image_key and image_key in IMAGE_CACHE:
                        cached_path = IMAGE_CACHE[image_key]
                        logger.debug(f"[{request_id}] Using cached image path for item {i+1}: {cached_path}")
                        image_paths.append(cached_path)
                        image = True
                        continue
                    
                    # Загружаем изображение, если оно не в кэше
                    logger.debug(f"[{request_id}] Processing image URL in item {i+1}: {image_url[:30]}...")
                    
                    # Загружаем изображение
                    image_path = retry_image_upload(image_url, api_key, request_id=request_id)
                    
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
                        logger.debug(f"[{request_id}] Image {i+1} successfully processed: {image_path}")
                    else:
                        logger.error(f"[{request_id}] Failed to upload image {i+1}")
            
            # Заменяем user_input текстовой частью, только если она не пуста
            if combined_text:
                user_input = combined_text
        
        # Подсчет токенов
        prompt_token = calculate_token(str(all_messages))
        
        # Проверка модели
        if PERMIT_MODELS_FROM_SUBSET_ONLY and model not in AVAILABLE_MODELS:
            return ERROR_HANDLER(1002, model)
        
        logger.debug(f"[{request_id}] Processing {prompt_token} prompt tokens with model {model}")
        
        # Проверка наличия web search
        web_search = request_data.get('web_search', False)
        num_of_site = request_data.get('num_of_site', 3)
        max_word = request_data.get('max_word', 500)
        
        # Формирование payload на основе наличия изображений
        if not image or not image_paths:
            logger.debug(f"[{request_id}] Creating CHAT_WITH_AI request")
            payload = {
                "type": "CHAT_WITH_AI",
                "model": model,
                "promptObject": {
                    "prompt": all_messages,
                    "isMixed": False,
                    "webSearch": web_search,
                    "numOfSite": num_of_site if web_search else None,
                    "maxWord": max_word if web_search else None
                }
            }
        else:
            logger.debug(f"[{request_id}] Creating CHAT_WITH_IMAGE request with {len(image_paths)} images: {image_paths}")
            
            # Добавляем приватность для распознавания людей, если это последнее сообщение
            if isinstance(user_input, str) and image:
                privacy_instructions = """
                Опиши изображение в общих чертах. Если на изображении есть люди, опиши их в общих чертах 
                (например, "человек в синем", "группа людей"), но не пытайся определить конкретных людей или их личности.
                Опиши композицию, объекты, цвета, обстановку и действия без идентификации конкретных лиц.
                Если на изображении есть текст, прочитай его, если это возможно.
                """
                all_messages += f"\n\n{privacy_instructions}\n\n"
            
            payload = {
                "type": "CHAT_WITH_IMAGE",
                "model": model,
                "promptObject": {
                    "prompt": all_messages,
                    "isMixed": False,
                    "imageList": image_paths,
                    "webSearch": web_search,
                    "numOfSite": num_of_site if web_search else None,
                    "maxWord": max_word if web_search else None
                }
            }
        
        headers = {"API-KEY": api_key, 'Content-Type': 'application/json'}
        
        # Выполнение запроса в зависимости от stream
        if not request_data.get('stream', False):
            # Обычный запрос
            logger.debug(f"[{request_id}] Sending non-streaming request to {ONE_MIN_API_URL}")
            
            try:
                response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
                logger.debug(f"[{request_id}] Response status code: {response.status_code}")
                
                if response.status_code != 200:
                    if response.status_code == 401:
                        return ERROR_HANDLER(1020, key=api_key)
                    try:
                        error_content = response.json()
                        logger.error(f"[{request_id}] Error response: {error_content}")
                    except:
                        logger.error(f"[{request_id}] Could not parse error response as JSON")
                    return ERROR_HANDLER(response.status_code)
                
                one_min_response = response.json()
                transformed_response = transform_response(one_min_response, request_data, prompt_token)
                
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
                response_stream = session.post(streaming_url, json=payload, headers=headers, stream=True)
                
                logger.debug(f"[{request_id}] Streaming response status code: {response_stream.status_code}")
                
                if response_stream.status_code != 200:
                    if response_stream.status_code == 401:
                        session.close()
                        return ERROR_HANDLER(1020, key=api_key)
                    
                    logger.error(f"[{request_id}] Error status code: {response_stream.status_code}")
                    try:
                        error_content = response_stream.json()
                        logger.error(f"[{request_id}] Error response: {error_content}")
                    except:
                        logger.error(f"[{request_id}] Could not parse error response as JSON")
                    
                    session.close()
                    return ERROR_HANDLER(response_stream.status_code)
                
                # Передаем сессию в generator
                return Response(
                    stream_response(response_stream, request_data, model, prompt_token, session),
                    content_type='text/event-stream'
                )
            except Exception as e:
                logger.error(f"[{request_id}] Exception during streaming request: {str(e)}")
                return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"[{request_id}] Exception during conversation processing: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error during conversation processing: {str(e)}"}), 500

@app.route('/v1/images/generations', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def generate_image():
    if request.method == 'OPTIONS':
        return handle_options_request()

    # Создаем уникальный ID для запроса
    request_id = str(uuid.uuid4())
    logger.debug(f"[{request_id}] Processing image generation request")

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
    headers = {'API-KEY': api_key, 'Content-Type': 'application/json'}
    
    request_data = request.json
    model = request_data.get('model', 'dall-e-2').strip()
    logger.debug(f"[{request_id}] Using model: {model}")
    
    # Преобразование параметров OpenAI в формат 1min.ai
    prompt = request_data.get('prompt', '')
    
    if model == 'dall-e-3':
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dall-e-3",
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get('n', 1),
                "size": request_data.get('size', '1024x1024'),
                "quality": request_data.get('quality', 'hd'),
                "style": request_data.get('style', 'vivid')
            }
        }
    elif model == 'dall-e-2':
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dall-e-2",
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get('n', 1),
                "size": request_data.get('size', '1024x1024')
            }
        }
    elif model == 'stable-diffusion-xl-1024-v1-0':
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "stable-diffusion-xl-1024-v1-0",
            "promptObject": {
                "prompt": prompt,
                "samples": request_data.get('n', 1),
                "size": request_data.get('size', '1024x1024'),
                "cfg_scale": request_data.get('cfg_scale', 7),
                "clip_guidance_preset": request_data.get('clip_guidance_preset', 'NONE'),
                "seed": request_data.get('seed', 0),
                "steps": request_data.get('steps', 30)
            }
        }
    elif model == 'midjourney':
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "midjourney",
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get('n', 1),
                "aspect_ratio": request_data.get('size', '1:1')
            }
        }
    else:
        logger.error(f"[{request_id}] Invalid model: {model}")
        return ERROR_HANDLER(1002, model)
    
    try:
        logger.debug(f"[{request_id}] Sending image generation request to {ONE_MIN_API_URL}")
        logger.debug(f"[{request_id}] Payload: {json.dumps(payload)[:200]}...")
        
        response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
        logger.debug(f"[{request_id}] Image generation response status code: {response.status_code}")
        
        if response.status_code != 200:
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            return jsonify({"error": response.json().get('error', 'Unknown error')}), response.status_code
        
        one_min_response = response.json()
        
        # Преобразование ответа 1min.ai в формат OpenAI
        try:
            image_url = one_min_response.get('aiRecord', {}).get('aiRecordDetail', {}).get('resultObject', [""])[0]
            
            if not image_url:
                # Попробуем другие пути извлечения URL
                if 'resultObject' in one_min_response:
                    image_url = one_min_response['resultObject'][0] if isinstance(one_min_response['resultObject'], list) else one_min_response['resultObject']
                    
            if not image_url:
                logger.error(f"[{request_id}] Could not extract image URL from API response")
                return jsonify({"error": "Could not extract image URL from API response"}), 500
                
            logger.debug(f"[{request_id}] Successfully generated image: {image_url[:50]}...")
            
            openai_response = {
                "created": int(time.time()),
                "data": [
                    {
                        "url": image_url
                    }
                ]
            }
            
            response = make_response(jsonify(openai_response))
            set_response_headers(response)
            return response, 200
        except Exception as e:
            logger.error(f"[{request_id}] Error processing image generation response: {str(e)}")
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"[{request_id}] Exception during image generation request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/images/variations', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def image_variations():
    if request.method == 'OPTIONS':
        return handle_options_request()

    # Создаем уникальный ID для запроса
    request_id = str(uuid.uuid4())
    logger.debug(f"[{request_id}] Processing image variation request")

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
    
    # Получение файла изображения
    if 'image' not in request.files:
        logger.error(f"[{request_id}] No image file provided")
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    model = request.form.get('model', 'dall-e-2').strip()
    n = request.form.get('n', 1)
    size = request.form.get('size', '1024x1024')
    
    logger.debug(f"[{request_id}] Using model: {model} for image variations")
    
    try:
        # Создаем новую сессию для загрузки изображения
        session = create_session()
        headers = {'API-KEY': api_key}
        
        # Загрузка изображения в 1min.ai
        files = {
            'asset': (image_file.filename, image_file, 'image/png')
        }
        
        try:
            asset_response = session.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
            logger.debug(f"[{request_id}] Image upload response status code: {asset_response.status_code}")
            
            if asset_response.status_code != 200:
                session.close()
                return jsonify({"error": asset_response.json().get('error', 'Failed to upload image')}), asset_response.status_code
            
            image_path = asset_response.json()['fileContent']['path']
            logger.debug(f"[{request_id}] Successfully uploaded image: {image_path}")
        finally:
            session.close()
        
        # Создание вариации в зависимости от модели
        if model == 'dall-e-2':
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "dall-e-2",
                "promptObject": {
                    "imageUrl": image_path,
                    "n": int(n),
                    "size": size
                }
            }
        elif model == 'clipdrop':
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "clipdrop",
                "promptObject": {
                    "imageUrl": image_path
                }
            }
        elif model == 'midjourney':
            payload = {
                "type": "IMAGE_VARIATOR",
                "model": "midjourney",
                "promptObject": {
                    "imageUrl": image_path,
                    "mode": "fast",
                    "n": int(n),
                    "isNiji6": False,
                    "aspect_width": 1,
                    "aspect_height": 1,
                    "maintainModeration": True
                }
            }
        else:
            logger.error(f"[{request_id}] Invalid model for variations: {model}")
            return ERROR_HANDLER(1002, model)
        
        headers['Content-Type'] = 'application/json'
        logger.debug(f"[{request_id}] Sending image variation request with payload: {json.dumps(payload)[:200]}...")
        
        response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
        logger.debug(f"[{request_id}] Image variation response status code: {response.status_code}")
        
        if response.status_code != 200:
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            logger.error(f"[{request_id}] Error in variation response: {response.text[:200]}")
            return jsonify({"error": response.json().get('error', 'Unknown error')}), response.status_code
        
        one_min_response = response.json()
        
        # Преобразование ответа 1min.ai в формат OpenAI
        try:
            # Безопасное извлечение URL изображения
            image_url = one_min_response.get('aiRecord', {}).get('aiRecordDetail', {}).get('resultObject', [""])[0]
            
            if not image_url:
                # Попробуем другие пути извлечения URL
                if 'resultObject' in one_min_response:
                    image_url = one_min_response['resultObject'][0] if isinstance(one_min_response['resultObject'], list) else one_min_response['resultObject']
                    
            if not image_url:
                logger.error(f"[{request_id}] Could not extract variation image URL from API response")
                return jsonify({"error": "Could not extract image URL from API response"}), 500
                
            logger.debug(f"[{request_id}] Successfully generated image variation: {image_url[:50]}...")
            
            openai_response = {
                "created": int(time.time()),
                "data": [
                    {
                        "url": image_url
                    }
                ]
            }
            
            response = make_response(jsonify(openai_response))
            set_response_headers(response)
            return response, 200
        except Exception as e:
            logger.error(f"[{request_id}] Error processing image variation response: {str(e)}")
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"[{request_id}] Exception during image variation request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/v1/keyword_research', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def keyword_research():
    if request.method == 'OPTIONS':
        return handle_options_request()

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
    headers = {'API-KEY': api_key, 'Content-Type': 'application/json'}
    
    request_data = request.json
    prompt = request_data.get('prompt', '')
    model = request_data.get('model', 'gpt-4o-mini')
    
    payload = {
        "type": "KEYWORD_RESEARCH",
        "model": model,
        "conversationId": "KEYWORD_RESEARCH",
        "promptObject": {
            "researchType": request_data.get('research_type', 'KEYWORD_STATISTICS'),
            "numberOfWord": request_data.get('number_of_words', 5),
            "prompt": prompt
        }
    }
    
    response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
    
    if response.status_code != 200:
        if response.status_code == 401:
            return ERROR_HANDLER(1020, key=api_key)
        return jsonify({"error": response.json().get('error', 'Unknown error')}), response.status_code
    
    one_min_response = response.json()
    
    try:
        result = one_min_response['aiRecord']['aiRecordDetail']['resultObject']
        
        openai_response = {
            "id": f"keyword-{uuid.uuid4()}",
            "created": int(time.time()),
            "model": model,
            "result": result
        }
        
        response = make_response(jsonify(openai_response))
        set_response_headers(response)
        return response, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/content/generate', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def generate_content():
    if request.method == 'OPTIONS':
        return handle_options_request()

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
    headers = {'API-KEY': api_key, 'Content-Type': 'application/json'}
    
    request_data = request.json
    prompt = request_data.get('prompt', '')
    model = request_data.get('model', 'gpt-4o-mini')
    
    payload = {
        "type": "CONTENT_GENERATOR_BLOG_ARTICLE",
        "model": model,
        "conversationId": "CONTENT_GENERATOR_BLOG_ARTICLE",
        "promptObject": {
            "language": request_data.get('language', 'English'),
            "tone": request_data.get('tone', 'informative'),
            "numberOfWord": request_data.get('number_of_words', 500),
            "numberOfSection": request_data.get('number_of_sections', 3),
            "keywords": request_data.get('keywords', ''),
            "prompt": prompt
        }
    }
    
    response = requests.post(ONE_MIN_API_URL, json=payload, headers=headers)
    
    if response.status_code != 200:
        if response.status_code == 401:
            return ERROR_HANDLER(1020, key=api_key)
        return jsonify({"error": response.json().get('error', 'Unknown error')}), response.status_code
    
    one_min_response = response.json()
    
    try:
        content = one_min_response['aiRecord']['aiRecordDetail']['resultObject']
        
        openai_response = {
            "id": f"content-{uuid.uuid4()}",
            "created": int(time.time()),
            "model": model,
            "content": content
        }
        
        response = make_response(jsonify(openai_response))
        set_response_headers(response)
        return response, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/assistants', methods=['POST', 'OPTIONS'])
@limiter.limit("500 per minute")
def create_assistant():
    if request.method == 'OPTIONS':
        return handle_options_request()

    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return ERROR_HANDLER(1021)
    
    api_key = auth_header.split(" ")[1]
    headers = {'API-KEY': api_key, 'Content-Type': 'application/json'}
    
    request_data = request.json
    name = request_data.get('name', 'PDF Assistant')
    instructions = request_data.get('instructions', '')
    model = request_data.get('model', 'gpt-4o-mini')
    file_ids = request_data.get('file_ids', [])
    
    # Создание беседы с PDF в 1min.ai
    payload = {
        "title": name,
        "type": "CHAT_WITH_PDF",
        "model": model,
        "fileList": file_ids
    }
    
    response = requests.post(ONE_MIN_CONVERSATION_API_URL, json=payload, headers=headers)
    
    if response.status_code != 200:
        if response.status_code == 401:
            return ERROR_HANDLER(1020, key=api_key)
        return jsonify({"error": response.json().get('error', 'Unknown error')}), response.status_code
    
    one_min_response = response.json()
    
    try:
        conversation_id = one_min_response.get('id')
        
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
            "metadata": {}
        }
        
        response = make_response(jsonify(openai_response))
        set_response_headers(response)
        return response, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def handle_options_request():
    response = make_response()
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response, 204

def transform_response(one_min_response, request_data, prompt_token):
    try:
        # Вывод структуры ответа для отладки
        logger.debug(f"Response structure: {json.dumps(one_min_response)[:200]}...")
        
        # Получаем ответ из соответствующего места в JSON
        result_text = one_min_response.get('aiRecord', {}).get("aiRecordDetail", {}).get("resultObject", [""])[0]
        
        if not result_text:
            # Альтернативные пути извлечения ответа
            if 'resultObject' in one_min_response:
                result_text = one_min_response['resultObject'][0] if isinstance(one_min_response['resultObject'], list) else one_min_response['resultObject']
            elif 'result' in one_min_response:
                result_text = one_min_response['result']
            else:
                # Если не нашли ответ по известным путям, возвращаем ошибку
                logger.error(f"Cannot extract response text from API result")
                result_text = "Error: Could not extract response from API"
        
        completion_token = calculate_token(result_text)
        logger.debug(f"Finished processing Non-Streaming response. Completion tokens: {str(completion_token)}")
        logger.debug(f"Total tokens: {str(completion_token + prompt_token)}")
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get('model', 'mistral-nemo').strip(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result_text,
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token,
                "completion_tokens": completion_token,
                "total_tokens": prompt_token + completion_token
            }
        }
    except Exception as e:
        logger.error(f"Error in transform_response: {str(e)}")
        # Возвращаем ошибку в формате, совместимом с OpenAI
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get('model', 'mistral-nemo').strip(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Error processing response: {str(e)}",
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token,
                "completion_tokens": 0,
                "total_tokens": prompt_token
            }
        }
    
def set_response_headers(response):
    response.headers['Content-Type'] = 'application/json'
    response.headers['Access -Control-Allow-Origin'] = '*'
    response.headers['X-Request-ID'] = str (uuid.uuid4())

def stream_response(response, request_data, model, prompt_tokens, session=None):
    all_chunks = ""
    
    try:
        for line in response.iter_lines():
            if not line:
                continue
            
            # Трассировка для отладки
            logger.debug(f"Received line from stream: {line[:100]}")
            
            try:
                if line.startswith(b'data: '):
                    data_str = line[6:].decode('utf-8')
                    if data_str == '[DONE]':
                        break
                    
                    # Проверяем, является ли это JSON-объектом
                    try:
                        json_data = json.loads(data_str)
                        # Если это полный ответ 1min.ai
                        if 'aiRecord' in json_data:
                            result_text = json_data.get('aiRecord', {}).get('aiRecordDetail', {}).get('resultObject', [""])[0]
                            chunk_text = result_text
                        else:
                            # Если это просто фрагмент текста в JSON
                            chunk_text = json_data.get('text', data_str)
                    except json.JSONDecodeError:
                        # Если это не JSON, используем как есть
                        chunk_text = data_str
                else:
                    # Для обычного текста
                    try:
                        # Проверяем, не является ли это JSON-объектом
                        raw_text = line.decode('utf-8')
                        try:
                            json_data = json.loads(raw_text)
                            # Если это полный ответ 1min.ai
                            if 'aiRecord' in json_data:
                                result_text = json_data.get('aiRecord', {}).get('aiRecordDetail', {}).get('resultObject', [""])[0]
                                chunk_text = result_text
                            else:
                                # Если это просто фрагмент текста в JSON
                                chunk_text = json_data.get('text', raw_text)
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
                            "delta": {
                                "content": chunk_text
                            },
                            "finish_reason": None
                        }
                    ]
                }
                
                yield f"data: {json.dumps(return_chunk)}\n\n"
                
            except Exception as e:
                logger.error(f"Error processing stream chunk: {str(e)}")
                continue
        
        # Подсчитываем токены после завершения
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
    
    finally:
        # Закрываем сессию, если она была передана
        if session:
            try:
                session.close()
                logger.debug("Streaming session closed properly")
            except Exception as e:
                logger.error(f"Error closing streaming session: {str(e)}")

def retry_image_upload(image_url, api_key, request_id=None):
    """Загружает изображение с повторными попытками, возвращает прямую ссылку на него"""
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Uploading image: {image_url}")
    
    # Создаем новую сессию для этого запроса
    session = create_session()
    
    try:
        # Загружаем изображение
        if image_url.startswith(('http://', 'https://')):
            # Загрузка по URL
            logger.debug(f"[{request_id}] Fetching image from URL: {image_url}")
            response = session.get(image_url, stream=True)
            response.raise_for_status()
            image_data = response.content
        else:
            # Декодирование base64
            logger.debug(f"[{request_id}] Decoding base64 image")
            image_data = base64.b64decode(image_url.split(',')[1])
        
        # Проверяем и конвертируем формат WebP при необходимости
        try:
            with BytesIO(image_data) as img_io:
                img = Image.open(img_io)
                
                # Логируем информацию о формате изображения
                logger.info(f"[{request_id}] Image format: {img.format}, Size: {img.size}, Mode: {img.mode}")
                
                # Конвертируем WebP и другие проблемные форматы в JPG
                if img.format in ['WEBP', 'GIF', 'ICO']:
                    logger.info(f"[{request_id}] Converting {img.format} to JPG")
                    if img.mode in ['RGBA', 'LA'] or (img.mode == 'P' and 'transparency' in img.info):
                        # Если есть прозрачность, сохраняем её, используя белый фон
                        background = Image.new('RGBA', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                        img = background.convert('RGB')
                    else:
                        img = img.convert('RGB')
                    
                    # Сохраняем преобразованное изображение
                    output = BytesIO()
                    img.save(output, format='JPEG', quality=95)
                    image_data = output.getvalue()
        except Exception as e:
            logger.error(f"[{request_id}] Error converting image: {str(e)}")
            # Продолжаем с оригинальными данными если конвертация не удалась
        
        # Проверяем размер файла
        if len(image_data) == 0:
            logger.error(f"[{request_id}] Empty image data")
            return None
        
        if len(image_data) > 5 * 1024 * 1024:  # 5MB
            logger.warning(f"[{request_id}] Image too large: {len(image_data) / (1024 * 1024):.2f}MB")
            
            # Пытаемся сжать изображение
            try:
                img = Image.open(BytesIO(image_data))
                img_io = BytesIO()
                img = img.convert('RGB')
                
                # Расчет нового размера с сохранением соотношения сторон
                ratio = min(1500 / img.width, 1500 / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                
                img.save(img_io, format='JPEG', quality=85)
                image_data = img_io.getvalue()
                logger.info(f"[{request_id}] Compressed image to {len(image_data) / (1024 * 1024):.2f}MB")
            except Exception as e:
                logger.error(f"[{request_id}] Error compressing image: {str(e)}")
                # Если сжатие не удалось, продолжаем с оригинальным
        
        # Создаем временный файл для загрузки
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        filename = f"temp_image_{random_string}.jpg"
        
        with open(filename, 'wb') as f:
            f.write(image_data)
        
        # Проверяем, что файл не пуст
        if os.path.getsize(filename) == 0:
            logger.error(f"[{request_id}] Empty image file created: {filename}")
            os.remove(filename)
            return None
        
        # Загружаем на сервер
        try:
            with open(filename, 'rb') as f:
                upload_response = session.post(
                    ONE_MIN_ASSET_URL,
                    headers={'API-KEY': api_key},
                    files={'asset': (filename, f, 'image/jpeg')}
                )
                
                if upload_response.status_code != 200:
                    logger.error(f"[{request_id}] Upload failed with status {upload_response.status_code}: {upload_response.text}")
                    return None
                
                # Получаем URL изображения
                upload_data = upload_response.json()
                if isinstance(upload_data, str):
                    try:
                        upload_data = json.loads(upload_data)
                    except:
                        logger.error(f"[{request_id}] Failed to parse upload response: {upload_data}")
                        return None
                
                logger.debug(f"[{request_id}] Upload response: {upload_data}")
                
                # Ищем URL в разных местах ответа
                url = None
                if isinstance(upload_data, dict):
                    # Специфичный формат для ONE_MIN_ASSET_URL
                    if 'fileContent' in upload_data and 'path' in upload_data['fileContent']:
                        url = upload_data['fileContent']['path']
                    else:
                        url = (upload_data.get('url') or 
                               upload_data.get('file_url') or 
                               (upload_data.get('data', {}) or {}).get('url') or
                               (upload_data.get('data', {}) or {}).get('file_url'))
                
                if url:
                    logger.info(f"[{request_id}] Image uploaded: {url}")
                    return url
                else:
                    logger.error(f"[{request_id}] No URL found in upload response")
                    return None
                    
        except Exception as e:
            logger.error(f"[{request_id}] Exception during image upload: {str(e)}")
            return None
        finally:
            # Удаляем временный файл
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to remove temp file: {str(e)}")
            
    except Exception as e:
        logger.error(f"[{request_id}] Exception during image processing: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # Закрываем сессию
        session.close()

# Создаем пул сессий для HTTP-запросов
def create_session():
    """Создает новую сессию с оптимальными настройками для API-запросов"""
    session = requests.Session()
    
    # Настройка повторных попыток для всех запросов
    retry_strategy = requests.packages.urllib3.util.retry.Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

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

if __name__ == '__main__':
    internal_ip = socket.gethostbyname(socket.gethostname())
    response = requests.get('https://api.ipify.org')
    public_ip = response.text
    logger.info(f"""{printedcolors.Color.fg.lightcyan}  
Server is ready to serve at:
Internal IP: {internal_ip}:5001
Public IP: {public_ip} (only if you've setup port forwarding on your router.)
Enter this url to OpenAI clients supporting custom endpoint:
{internal_ip}:5001/v1
If does not work, try:
{internal_ip}:5001/v1/chat/completions
{printedcolors.Color.reset}""")
    serve(app, host='0.0.0.0', port=5001, threads=6) # Thread has a default of 4 if not specified. We use 6 to increase performance and allow multiple requests at once.
