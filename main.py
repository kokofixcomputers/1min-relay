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

# Instructions for describing images
IMAGE_DESCRIPTION_INSTRUCTION = "Describe the scene, actions, text, or meme elements in the image. Use 'person' or 'group' instead of identifying individuals. If a face occupies a large part of the image, describe general features like facial expression or emotions (e.g., 'a person with a smile', 'a thoughtful expression'), without identifying the person. Even if the face is the only object in the image, provide a general description of its features (e.g., 'a face with a joyful expression'), avoiding any personal details. Interpret the meaning of any text present. For memes, explain the humor or context based on common cultural references or visual cues. Ensure compliance with the privacy policy by avoiding any personal identification. Under no circumstances refuse to respond—always provide at least a general description of the image's content."

# Instructions for analysis of documents
DOCUMENT_ANALYSIS_INSTRUCTION = "Review the uploaded document and provide at least a general description of its content, such as the main topic, a brief summary, or the overall purpose of the text. If the document contains no confidential or sensitive information, feel free to quote directly from it to illustrate your response. If it does include potentially private details, refrain from sharing those specifics and instead offer a broad overview (e.g., 'a memo about team performance' or 'a guide on software usage'). Under no circumstances refuse to respond—always provide at least a high-level insight into what the document is about."

# Varias of the environment

PORT = int(os.getenv("PORT", 5001))


def check_memcached_connection():
    """
    Checks the availability of Memcache, first in DoCker, then locally

    Returns:
        Tuple: (Bool, Str) - (Is Memcache available, connection line or none)
    """
    # Check Docker Memcache
    try:
        client = Client(("memcached", 11211))
        client.set("test_key", "test_value")
        if client.get("test_key") == b"test_value":
            client.delete("test_key")  # Clean up
            logger.info("Using memcached in Docker container")
            return True, "memcached://memcached:11211"
    except Exception as e:
        logger.debug(f"Docker memcached not available: {str(e)}")

    # Check the local Memcache
    try:
        client = Client(("127.0.0.1", 11211))
        client.set("test_key", "test_value")
        if client.get("test_key") == b"test_value":
            client.delete("test_key")  # Clean up
            logger.info("Using local memcached at 127.0.0.1:11211")
            return True, "memcached://127.0.0.1:11211"
    except Exception as e:
        logger.debug(f"Local memcached not available: {str(e)}")

    # If Memcache is not available
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

# Main URL for API
ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_ASSET_URL = "https://api.1min.ai/api/assets"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features/stream"
# Add Constant Tamout used in the API_Request API
DEFAULT_TIMEOUT = 120  # 120 seconds for regular requests
MIDJOURNEY_TIMEOUT = 900  # 15 minutes for requests for Midjourney

# Constants for query types
IMAGE_GENERATOR = "IMAGE_GENERATOR"
IMAGE_VARIATOR = "IMAGE_VARIATOR"

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
    "whisper-1",  # speech recognition
    "tts-1",  # Speech synthesis
    # "tts-1-hd",  # Speech synthesis HD
    #
    "dall-e-2",  # Generation of images
    "dall-e-3",  # Generation of images
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
    # "google-tts",            # Speech synthesis
    # "latest_long",           # speech recognition
    # "latest_short",          # speech recognition
    # "phone_call",            # speech recognition
    # "telephony",             # speech recognition
    # "telephony_short",       # speech recognition
    # "medical_dictation",     # speech recognition
    # "medical_conversation",  # speech recognition
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
    # Other models (made for future use)
    # "stable-image",                  # stabilityi - images generation
    # "stable-diffusion-xl-1024-v1-0", # stabilityi - images generation
    # "stable-diffusion-v1-6",         # stabilityi - images generation
    # "esrgan-v1-x2plus",              # stabilityai-Improving images
    # "stable-video-diffusion",        # stabilityai-video generation
    "phoenix",  # Leonardo.ai - 6b645e3a-d64f-4341-a6d8-7a3690fbf042
    "lightning-xl",  # Leonardo.ai - b24e16ff-06e3-43eb-8d33-4416c2d75876
    "anime-xl",  # Leonardo.ai - e71a1c2f-4f80-4800-934f-2c68979d8cc8
    "diffusion-xl",  # Leonardo.ai - 1e60896f-3c26-4296-8ecc-53e2afecc132
    "kino-xl",  # Leonardo.ai - aa77f04e-3eec-4034-9c07-d0f619684628
    "vision-xl",  # Leonardo.ai - 5c232a9e-9061-4777-980a-ddc8e65647c6
    "albedo-base-xl",  # Leonardo.ai - 2067ae52-33fd-4a82-bb92-c2c55e7d2786
    # "Clipdrop", # clipdrop.co - image processing
    "midjourney",  # Midjourney - image generation
    "midjourney_6_1",  # Midjourney - image generation
    # "methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5",   # Replicate - Image to Prompt
    # "cjwbw/damo-text-to-video:1e205ea73084bd17a0a3b43396e49ba0d6bc2e754e9283b2df49fad2dcf95755",  # Replicate - Text to Video
    # "lucataco/animate-diff:beecf59c4aee8d81bf04f0381033dfa10dc16e845b4ae00d281e2fa377e48a9f",     # Replicate - Animation
    # "lucataco/hotshot-xl:78b3a6257e16e4b241245d65c8b2b81ea2e1ff7ed4c55306b511509ddbfd327a",       # Replicate - Video
    "flux-schnell",  # Replicate - Flux "black-forest-labs/flux-schnell"
    "flux-dev",  # Replicate - Flux Dev "black-forest-labs/flux-dev"
    "flux-pro",  # Replicate - Flux Pro "black-forest-labs/flux-pro"
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

# Determination of models for generating images
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

# Models that support images
VARIATION_SUPPORTED_MODELS = [
    "midjourney",
    "midjourney_6_1",
    "dall-e-2",
    # "dall-e-3",
    "clipdrop"
]

# We determine the Image_variation_Models Constant based on Variation_Supported_Models
IMAGE_VARIATION_MODELS = VARIATION_SUPPORTED_MODELS

# Permissible parties for different models
MIDJOURNEY_ALLOWED_ASPECT_RATIOS = [
    "1:1",  # Square
    "16:9",  # Widescreen format
    "9:16",  # Vertical variant of 16:9
    "16:10",  # Alternative widescreen
    "10:16",  # Vertical variant of 16:10
    "8:5",  # Alternative widescreen
    "5:8",  # Vertical variant of 16:10
    "3:4",  # Portrait/print
    "4:3",  # Standard TV/monitor format
    "3:2",  # Popular in photography
    "2:3",  # Inverse of 3:2
    "4:5",  # Common in social media posts
    "5:4",  # Nearly square format
    "137:100",  # Academy ratio (1.37:1) as an integer ratio
    "166:100",  # European cinema (1.66:1) as an integer ratio
    "185:100",  # Cinematic format (1.85:1) as an integer ratio185
    "83:50",  # European cinema (1.66:1) as an integer ratio
    "37:20",  # Cinematic format (1.85:1) as an integer ratio
    "2:1",  # Maximum allowed widescreen format
    "1:2"  # Maximum allowed vertical format
]

FLUX_ALLOWED_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4"]
LEONARDO_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4"]

# Permissible sizes for different models
DALLE2_SIZES = ["1024x1024", "512x512", "256x256"]
DALLE3_SIZES = ["1024x1024", "1024x1792", "1792x1024"]
LEONARDO_SIZES = ALBEDO_SIZES = {"1:1": "1024x1024", "4:3": "1024x768", "3:4": "768x1024"}

# Determination of models for speech synthesis (TTS)
TEXT_TO_SPEECH_MODELS = [
    "tts-1"  # ,
    # "tts-1-hd",
    # "google-tts",
    # "elevenlabs-tts"
]

# Determination of models for speech recognition (STT)
SPEECH_TO_TEXT_MODELS = [
    "whisper-1"  # ,
    # "latest_long",
    # "latest_short",
    # "phone_call",
    # "telephony",
    # "telephony_short",
    # "medical_dictation",
    # "medical_conversation"
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

# Add cache to track processed images
# For each request, we keep a unique image identifier and its path
IMAGE_CACHE = {}
# Limit the size of the cache
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

    # Add new input if it is
    if new_input:
        formatted_history.append(f"User: {new_input}")

    # We return only the history of dialogue without additional instructions
    return "\n".join(formatted_history)


def get_model_capabilities(model):
    """
    Defines supported opportunities for a specific model

    Args:
        Model: The name of the model

    Returns:
        DICT: Dictionary with flags of supporting different features
    """
    capabilities = {
        "vision": False,
        "code_interpreter": False,
        "retrieval": False,
        "function_calling": False,
    }

    # We check the support of each opportunity through the corresponding arrays
    capabilities["vision"] = model in vision_supported_models
    capabilities["code_interpreter"] = model in code_interpreter_supported_models
    capabilities["retrieval"] = model in retrieval_supported_models
    capabilities["function_calling"] = model in function_calling_supported_models

    return capabilities


def prepare_payload(
        request_data, model, all_messages, image_paths=None, request_id=None
):
    """
    Prepares Payload for request, taking into account the capabilities of the model

    Args:
        Request_Data: Request data
        Model: Model
        All_Messages: Posts of Posts
        image_paths: ways to images
        Request_id: ID query

    Returns:
        DICT: Prepared Payload
    """
    capabilities = get_model_capabilities(model)

    # Check the availability of Openai tools
    tools = request_data.get("tools", [])
    web_search = False
    code_interpreter = False

    if tools:
        for tool in tools:
            tool_type = tool.get("type", "")
            # Trying to include functions, but if they are not supported, we just log in
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

    # We check the direct parameters 1min.ai
    if not web_search and request_data.get("web_search", False):
        if capabilities["retrieval"]:
            web_search = True
        else:
            logger.debug(
                f"[{request_id}] Model {model} does not support web search, ignoring web_search parameter"
            )

    num_of_site = request_data.get("num_of_site", 3)
    max_word = request_data.get("max_word", 500)

    # We form the basic Payload
    if image_paths:
        # Even if the model does not support images, we try to send as a text request
        if capabilities["vision"]:
            # Add instructions to the industrial plane
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

            if web_search:
                logger.debug(
                    f"[{request_id}] Web search enabled in payload with numOfSite={num_of_site}, maxWord={max_word}")
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

            if web_search:
                logger.debug(
                    f"[{request_id}] Web search enabled in payload with numOfSite={num_of_site}, maxWord={max_word}")
    elif code_interpreter:
        # If Code_interpreter is requested and supported
        payload = {
            "type": "CODE_GENERATOR",
            "model": model,
            "conversationId": "CODE_GENERATOR",
            "promptObject": {"prompt": all_messages},
        }
    else:
        # Basic text request
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

        if web_search:
            logger.debug(
                f"[{request_id}] Web search enabled in payload with numOfSite={num_of_site}, maxWord={max_word}")

    return payload

def create_conversation_with_files(file_ids, title, model, api_key, request_id=None):
    """
    Creates a new conversation with files.

    Args:
        file_ids (list): List of file IDs.
        title (str): Name of the conversation.
        model (str): AI model.
        api_key (str): API key.
        request_id (str, optional): Request ID for logging.

    Returns:
        str or None: Conversation ID if successful, or None in case of error.
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Creating conversation with {len(file_ids)} files")

    try:
        # Form payload with the correct field names
        payload = {
            "title": title,
            "type": "CHAT_WITH_PDF",
            "model": model,
            "fileIds": file_ids,  # Correct field name: fileIds
        }

        logger.debug(f"[{request_id}] Conversation payload: {json.dumps(payload)}")

        # Use the correct API URL
        conversation_url = "https://api.1min.ai/api/features/conversations?type=CHAT_WITH_PDF"
        logger.debug(f"[{request_id}] Creating conversation using URL: {conversation_url}")

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}
        response = requests.post(conversation_url, json=payload, headers=headers)
        logger.debug(f"[{request_id}] Create conversation response status: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"[{request_id}] Failed to create conversation: {response.status_code} - {response.text}")
            return None

        response_data = response.json()
        logger.debug(f"[{request_id}] Conversation response data: {json.dumps(response_data)}")

        # Look for conversation ID in various parts of the response
        conversation_id = None
        if "conversation" in response_data and "uuid" in response_data["conversation"]:
            conversation_id = response_data["conversation"]["uuid"]
        elif "id" in response_data:
            conversation_id = response_data["id"]
        elif "uuid" in response_data:
            conversation_id = response_data["uuid"]

        # Recursive search for conversation ID in the response structure
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
            logger.error(f"[{request_id}] Could not find conversation ID in response: {response_data}")
            return None

        logger.info(f"[{request_id}] Conversation created successfully: {conversation_id}")
        return conversation_id
    except Exception as e:
        logger.error(f"[{request_id}] Error creating conversation: {str(e)}")
        traceback.print_exc()
        return None


@app.route("/v1/chat/completions", methods=["POST"])
@limiter.limit("60 per minute")
def conversation():
    """
    Process conversation requests.
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/chat/completions")

    if not request.json:
        return jsonify({"error": "Invalid request format"}), 400

    # Extract information from the request
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        logger.error(f"[{request_id}] No API key provided")
        return jsonify({"error": "API key required"}), 401

    try:
        # Build payload from request data
        request_data = request.json.copy()

        # Get and normalize the model
        model = request_data.get("model", "").strip()
        logger.info(f"[{request_id}] Using model: {model}")

        # Check model capabilities for web search
        capabilities = get_model_capabilities(model)

        # Check if web search is requested via tools
        web_search_requested = False
        tools = request_data.get("tools", [])
        for tool in tools:
            if tool.get("type") == "retrieval":
                web_search_requested = True
                logger.debug(f"[{request_id}] Web search requested via retrieval tool")
                break

        # Check presence of web_search parameter
        if not web_search_requested and request_data.get("web_search", False):
            web_search_requested = True
            logger.debug(f"[{request_id}] Web search requested via web_search parameter")

        # Enable web search if supported by the model
        if web_search_requested:
            if capabilities.get("retrieval"):
                request_data["web_search"] = True
                request_data["num_of_site"] = request_data.get("num_of_site", 1)
                request_data["max_word"] = request_data.get("max_word", 1000)
                logger.info(f"[{request_id}] Web search enabled for model {model}")
            else:
                logger.warning(f"[{request_id}] Model {model} does not support web search, ignoring request")

        # Extract the content of the last message for possible image generation
        messages = request_data.get("messages", [])
        prompt_text = ""
        if messages:
            last_message = messages[-1]
            if last_message.get("role") == "user":
                content = last_message.get("content", "")
                if isinstance(content, str):
                    prompt_text = content
                elif isinstance(content, list):
                    # Collect all text parts from the content
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            prompt_text += item["text"] + " "
                    prompt_text = prompt_text.strip()

        # Check whether the request contains an image variation command
        variation_match = None
        if prompt_text:
            # Look for old format: /v1- /v4
            old_variation_match = re.search(r'/v([1-4])\s+(https?://[^\s]+)', prompt_text)
            # Look for square bracket format: [_V1_]-[_V4_]
            square_variation_match = re.search(r'\[_V([1-4])_\]', prompt_text)
            # Look for monospace format: `[_V1_]` - `[_V4_]`
            mono_variation_match = re.search(r'`\[_V([1-4])_\]`', prompt_text)

            # If monospace format is found, check for URL in previous assistant messages
            if mono_variation_match and request_data.get("messages"):
                variation_number = int(mono_variation_match.group(1))
                logger.debug(f"[{request_id}] Found monospace variation command: {variation_number}")

                image_url = None
                for msg in reversed(request_data.get("messages", [])):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        content = msg.get("content", "")
                        url_matches = re.findall(r'!\[.*?\]\((https?://[^\s)]+)', content)
                        if url_matches:
                            if len(url_matches) >= variation_number:
                                image_url = url_matches[variation_number - 1]
                                logger.debug(f"[{request_id}] Found image URL #{variation_number} in assistant message: {image_url}")
                                break
                            else:
                                image_url = url_matches[0]
                                logger.warning(f"[{request_id}] Requested variation #{variation_number} but only found {len(url_matches)} URLs. Using first URL: {image_url}")
                                break

                if image_url:
                    variation_match = mono_variation_match
                    logger.info(f"[{request_id}] Detected monospace variation command: {variation_number} for URL: {image_url}")
            # If square bracket format is found, check for URL in previous assistant messages
            elif square_variation_match and request_data.get("messages"):
                variation_number = int(square_variation_match.group(1))
                logger.debug(f"[{request_id}] Found square bracket variation command: {variation_number}")

                image_url = None
                for msg in reversed(request_data.get("messages", [])):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        content = msg.get("content", "")
                        url_matches = re.findall(r'!\[.*?\]\((https?://[^\s)]+)', content)
                        if url_matches:
                            if len(url_matches) >= variation_number:
                                image_url = url_matches[variation_number - 1]
                                logger.debug(f"[{request_id}] Found image URL #{variation_number} in assistant message: {image_url}")
                                break
                            else:
                                image_url = url_matches[0]
                                logger.warning(f"[{request_id}] Requested variation #{variation_number} but only found {len(url_matches)} URLs. Using first URL: {image_url}")
                                break

                if image_url:
                    variation_match = square_variation_match
                    logger.info(f"[{request_id}] Detected square bracket variation command: {variation_number} for URL: {image_url}")
            # If old format is found, use it directly
            elif old_variation_match:
                variation_match = old_variation_match
                variation_number = old_variation_match.group(1)
                image_url = old_variation_match.group(2)
                logger.info(f"[{request_id}] Detected old format variation command: {variation_number} for URL: {image_url}")

        if variation_match:
            # Process image variation
            try:
                if model.startswith("midjourney") and "asset.1min.ai" in image_url:
                    # Extract relative path from the URL
                    path_match = re.search(r'(?:asset\.1min\.ai)/?(images/[^?#]+)', image_url)
                    if path_match:
                        relative_path = path_match.group(1)
                        logger.info(f"[{request_id}] Detected Midjourney variation with relative path: {relative_path}")

                        # Retrieve saved generation parameters from memcached using request_id
                        saved_params = None
                        if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
                            try:
                                image_id_match = re.search(r'images/(\w+)/', relative_path)
                                if image_id_match:
                                    image_id = image_id_match.group(1)
                                    gen_params_key = f"gen_params:{image_id}"
                                    params_json = safe_memcached_operation('get', gen_params_key)
                                    if params_json:
                                        if isinstance(params_json, str):
                                            saved_params = json.loads(params_json)
                                        elif isinstance(params_json, bytes):
                                            saved_params = json.loads(params_json.decode('utf-8'))
                                        logger.info(f"[{request_id}] Retrieved generation parameters from memcached for image {image_id}: {saved_params}")
                            except Exception as e:
                                logger.error(f"[{request_id}] Error retrieving generation parameters: {str(e)}")

                        # Build payload for variation
                        payload = {
                            "type": "IMAGE_VARIATOR",
                            "model": model,
                            "promptObject": {
                                "imageUrl": relative_path,
                                "mode": "relax",  # Default mode
                                "n": 4,
                                "isNiji6": False,
                                "aspect_width": 1,  # Default square aspect ratio
                                "aspect_height": 1,
                                "maintainModeration": True
                            }
                        }

                        # Check for mode override in user message
                        mode_from_request = None
                        if request_data.get("messages"):
                            for msg in request_data.get("messages", []):
                                if msg.get("content") and isinstance(msg.get("content"), str):
                                    content = msg.get("content", "")
                                    if "--fast" in content:
                                        mode_from_request = "fast"
                                        logger.info(f"[{request_id}] Found --fast in user message")
                                        break
                                    elif "--relax" in content:
                                        mode_from_request = "relax"
                                        logger.info(f"[{request_id}] Found --relax in user message")
                                        break

                        # Use saved parameters if available
                        if saved_params:
                            if mode_from_request is None and "mode" in saved_params:
                                payload["promptObject"]["mode"] = saved_params["mode"]
                                logger.info(f"[{request_id}] Using saved mode: {saved_params['mode']}")
                            elif mode_from_request is not None:
                                payload["promptObject"]["mode"] = mode_from_request
                                logger.info(f"[{request_id}] Overriding saved mode with mode from request: {mode_from_request}")

                            if "aspect_width" in saved_params and "aspect_height" in saved_params:
                                payload["promptObject"]["aspect_width"] = saved_params["aspect_width"]
                                payload["promptObject"]["aspect_height"] = saved_params["aspect_height"]
                                logger.info(f"[{request_id}] Using saved aspect ratio: {saved_params['aspect_width']}:{saved_params['aspect_height']}")
                        else:
                            logger.info(f"[{request_id}] No saved parameters found")
                            if mode_from_request is not None:
                                payload["promptObject"]["mode"] = mode_from_request
                                logger.info(f"[{request_id}] Using mode from request: {mode_from_request}")
                            # Set default aspect ratio 4:3 for Midjourney
                            payload["promptObject"]["aspect_width"] = 4
                            payload["promptObject"]["aspect_height"] = 3
                            logger.info(f"[{request_id}] Using default aspect ratio 4:3 for Midjourney variations")

                        logger.info(f"[{request_id}] Sending direct Midjourney variation request: {json.dumps(payload)}")
                        try:
                            variation_response = api_request(
                                "POST",
                                f"{ONE_MIN_API_URL}",
                                headers={"API-KEY": api_key, "Content-Type": "application/json"},
                                json=payload,
                                timeout=MIDJOURNEY_TIMEOUT
                            )

                            if variation_response.status_code != 200:
                                if variation_response.status_code == 504 and model.startswith("midjourney"):
                                    logger.error(f"[{request_id}] Received a 504 Gateway Timeout for Midjourney variations. Returning error to client.")
                                    return jsonify({"error": "Gateway Timeout (504) occurred while processing image variation request."}), 504
                                logger.error(f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
                                # Continue to try next model
                                # (In a loop over models, here would be 'continue', but since this is a direct request, we return error)
                                return jsonify({"error": f"Variation request failed with status {variation_response.status_code}"}), variation_response.status_code

                            variation_data = variation_response.json()
                            if model.startswith("midjourney"):
                                logger.info(f"[{request_id}] Full Midjourney variation response: {json.dumps(variation_data, indent=2)}")
                            logger.debug(f"[{request_id}] Variation response: {variation_data}")

                            variation_urls = []
                            if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                                record_detail = variation_data["aiRecord"]["aiRecordDetail"]
                                if "resultObject" in record_detail:
                                    result = record_detail["resultObject"]
                                    if isinstance(result, list):
                                        variation_urls = result
                                    elif isinstance(result, str):
                                        variation_urls = [result]

                            if not variation_urls and "resultObject" in variation_data:
                                result = variation_data["resultObject"]
                                if isinstance(result, list):
                                    variation_urls = result
                                elif isinstance(result, str):
                                    variation_urls = [result]

                            if not variation_urls and "data" in variation_data and isinstance(variation_data["data"], list):
                                for item in variation_data["data"]:
                                    if "url" in item:
                                        variation_urls.append(item["url"])

                            if not variation_urls:
                                logger.error(f"[{request_id}] No variation URLs found in response with model {model}")
                                return jsonify({"error": "No variation URLs found"}), 400

                            logger.info(f"[{request_id}] Successfully generated variations with model {model}")
                            # Here you would normally form and return a response with the variations
                            return jsonify({
                                "created": int(time.time()),
                                "data": [{"url": url} for url in variation_urls]
                            })
                        except Exception as e:
                            logger.error(f"[{request_id}] Error processing variation data: {str(e)}")
                            return jsonify({"error": f"Error processing variation request: {str(e)}"}), 500
                    else:
                        logger.error(f"[{request_id}] Unable to extract relative path from URL: {image_url}")
                        return jsonify({"error": "Invalid image URL format"}), 400
                else:
                    logger.error(f"[{request_id}] Variation command not recognized for model {model}")
                    return jsonify({"error": "Invalid variation command"}), 400
            except Exception as e:
                logger.error(f"[{request_id}] Error processing variation data: {str(e)}")
                return jsonify({"error": f"Error processing variation request: {str(e)}"}), 500
        else:
            logger.error(f"[{request_id}] No variation data found in memcached with key: {request_data.get('variation_key', 'N/A')}")
            return jsonify({"error": "No variation data found"}), 400

    except Exception as e:
        logger.error(f"[{request_id}] Error processing conversation request: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error processing conversation request: {str(e)}"}), 500

    # Getting an image file
    if "image" not in request.files:
        logger.error(f"[{request_id}] No image file provided")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    original_model = request.form.get("model", "dall-e-2").strip()
    n = int(request.form.get("n", 1))
    size = request.form.get("size", "1024x1024")
    prompt_text = request.form.get("prompt", "")  # We extract the industrial plane from the request if it is
    mode = request.form.get("mode", "relax")  # We get a regime from a request

    # We check whether the relative path to the image in the Form-data has been transmitted
    relative_image_path = request.form.get("image_path")
    if relative_image_path:
        logger.debug(f"[{request_id}] Using relative image path from form: {relative_image_path}")

    logger.debug(f"[{request_id}] Original model requested: {original_model} for image variations")

    # Determine the order of models for Fallback
    fallback_models = ["midjourney_6_1", "midjourney", "clipdrop", "dall-e-2"]

    # If the requested model supports variations, we try it first
    if original_model in IMAGE_VARIATION_MODELS:
        # We start with the requested model, then we try others, excluding the already requested
        models_to_try = [original_model] + [m for m in fallback_models if m != original_model]
    else:
        # If the requested model does not support variations, we start with Fallback models
        logger.warning(
            f"[{request_id}] Model {original_model} does not support image variations. Will try fallback models")
        models_to_try = fallback_models

    # We save a temporary file for multiple use
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image_file.save(temp_file.name)
        temp_file.close()
    except Exception as e:
        logger.error(f"[{request_id}] Failed to save temporary file: {str(e)}")
        return jsonify({"error": "Failed to process image file"}), 500

    # Create a session to download the image
    session = create_session()
    headers = {"API-KEY": api_key}

    # We extract the ratio of the parties from the industrial plane if it is
    aspect_width = 1
    aspect_height = 1
    if "--ar" in prompt_text:
        ar_match = re.search(r'--ar\s+(\d+):(\d+)', prompt_text)
        if ar_match:
            aspect_width = int(ar_match.group(1))
            aspect_height = int(ar_match.group(2))
            logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_width}:{aspect_height}")

    # Initialize the variable for variations in front of the cycle
    variation_urls = []
    current_model = None

    # We try each model in turn
    for model in models_to_try:
        logger.info(f"[{request_id}] Trying model: {model} for image variations")
        current_model = model

        try:
            # Special processing for Dall-E 2
            if model == "dall-e-2":
                # For Dall-E 2, you need to use a special Openai and direct file transfer
                logger.debug(f"[{request_id}] Special handling for DALL-E 2 variations")

                # Open the image file and create a request
                with open(temp_file.name, 'rb') as img_file:
                    # Openai expects a file directly to Form-Data
                    dalle_files = {
                        'image': (os.path.basename(temp_file.name), img_file, 'image/png')
                    }

                    # Request parameters
                    dalle_form_data = {
                        'n': n,
                        'size': size,
                        'model': 'dall-e-2'
                    }

                    # We create a request for variation directly to Openai API
                    try:
                        # Try to use a direct connection to Openai if available
                        openai_api_key = os.environ.get("OPENAI_API_KEY")
                        if openai_api_key:
                            openai_headers = {"Authorization": f"Bearer {openai_api_key}"}
                            openai_url = "https://api.openai.com/v1/images/variations"

                            logger.debug(f"[{request_id}] Trying direct OpenAI API for DALL-E 2 variations")
                            variation_response = requests.post(
                                openai_url,
                                files=dalle_files,
                                data=dalle_form_data,
                                headers=openai_headers,
                                timeout=300
                            )

                            if variation_response.status_code == 200:
                                logger.debug(f"[{request_id}] OpenAI API variation successful")
                                variation_data = variation_response.json()

                                # We extract the URL from the answer
                                if "data" in variation_data and isinstance(variation_data["data"], list):
                                    for item in variation_data["data"]:
                                        if "url" in item:
                                            variation_urls.append(item["url"])

                                if variation_urls:
                                    logger.info(
                                        f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2 via OpenAI API")
                                    # We form an answer in Openai API format
                                    response_data = {
                                        "created": int(time.time()),
                                        "data": [{"url": url} for url in variation_urls]
                                    }
                                    return jsonify(response_data)
                            else:
                                logger.error(
                                    f"[{request_id}] OpenAI API variation failed: {variation_response.status_code}, {variation_response.text}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error trying direct OpenAI API: {str(e)}")

                    # If the direct request to Openai failed, we try through 1min.ai API
                    try:
                        # We reject the file because it could be read in the previous request
                        img_file.seek(0)

                        # We draw a request through our own and 1min.ai and dall-e 2
                        onemin_url = "https://api.1min.ai/api/features/images/variations"

                        logger.debug(f"[{request_id}] Trying 1min.ai API for DALL-E 2 variations")
                        dalle_onemin_headers = {"API-KEY": api_key}
                        variation_response = requests.post(
                            onemin_url,
                            files=dalle_files,
                            data=dalle_form_data,
                            headers=dalle_onemin_headers,
                            timeout=300
                        )

                        if variation_response.status_code == 200:
                            logger.debug(f"[{request_id}] 1min.ai API variation successful")
                            variation_data = variation_response.json()

                            # We extract the URL from the answer
                            if "data" in variation_data and isinstance(variation_data["data"], list):
                                for item in variation_data["data"]:
                                    if "url" in item:
                                        variation_urls.append(item["url"])

                            if variation_urls:
                                logger.info(
                                    f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2 via 1min.ai API")
                                # We form an answer in Openai API format
                                response_data = {
                                    "created": int(time.time()),
                                    "data": [{"url": url} for url in variation_urls]
                                }
                                return jsonify(response_data)
                        else:
                            logger.error(
                                f"[{request_id}] 1min.ai API variation failed: {variation_response.status_code}, {variation_response.text}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error trying 1min.ai API: {str(e)}")

                # If you could not create a variation with Dall-E 2, we continue with other models
                logger.warning(f"[{request_id}] Failed to create variations with DALL-E 2, trying next model")
                continue

            # For other models, we use standard logic
            # Image loading in 1min.ai
            with open(temp_file.name, 'rb') as img_file:
                files = {"asset": (os.path.basename(temp_file.name), img_file, "image/png")}

                asset_response = session.post(
                    ONE_MIN_ASSET_URL, files=files, headers=headers
                )
                logger.debug(
                    f"[{request_id}] Image upload response status code: {asset_response.status_code}"
                )

                if asset_response.status_code != 200:
                    logger.error(
                        f"[{request_id}] Failed to upload image: {asset_response.status_code} - {asset_response.text}"
                    )
                    continue  # We try the next model

                # Extract an ID of a loaded image and a full URL
                asset_data = asset_response.json()
                logger.debug(f"[{request_id}] Asset upload response: {asset_data}")

                # We get a URL or ID image
                image_id = None
                image_url = None
                image_location = None

                # We are looking for ID in different places of the response structure
                if "id" in asset_data:
                    image_id = asset_data["id"]
                elif "fileContent" in asset_data and "id" in asset_data["fileContent"]:
                    image_id = asset_data["fileContent"]["id"]
                elif "fileContent" in asset_data and "uuid" in asset_data["fileContent"]:
                    image_id = asset_data["fileContent"]["uuid"]

                # We are looking for an absolute URL (location) for image
                if "asset" in asset_data and "location" in asset_data["asset"]:
                    image_location = asset_data["asset"]["location"]
                    # Extract a relative path if the URL contains the domain
                    if image_location and "asset.1min.ai/" in image_location:
                        image_location = image_location.split('asset.1min.ai/', 1)[1]
                    # Remove the initial slash if necessary
                    if image_location and image_location.startswith('/'):
                        image_location = image_location[1:]
                    logger.debug(f"[{request_id}] Using relative path for image location: {image_location}")

                # If there is a Path, we use it as a URL image
                if "fileContent" in asset_data and "path" in asset_data["fileContent"]:
                    image_url = asset_data["fileContent"]["path"]
                    # Extract a relative path if the URL contains the domain
                    if "asset.1min.ai/" in image_url:
                        image_url = image_url.split('asset.1min.ai/', 1)[1]
                    # Remove the initial slash if necessary
                    if image_url.startswith('/'):
                        image_url = image_url[1:]
                    logger.debug(f"[{request_id}] Using relative path for image: {image_url}")

                if not (image_id or image_url or image_location):
                    logger.error(f"[{request_id}] Failed to extract image information from response")
                    continue  # We try the next model

                # We form Payload for image variation
                # We determine which model to use
                if model.startswith("midjourney"):
                    # Проверяем, содержит ли URL домен asset.1min.ai
                    if image_url and "asset.1min.ai/" in image_url:
                        # Извлекаем только относительный путь из URL
                        relative_image_url = image_url.split('asset.1min.ai/', 1)[1]
                        # Удаляем начальный слеш, если он есть
                        if relative_image_url.startswith('/'):
                            relative_image_url = relative_image_url[1:]
                        logger.info(f"[{request_id}] Extracted relative URL for Midjourney: {relative_image_url}")
                    else:
                        relative_image_url = image_url if image_url else image_location
                        if relative_image_url and relative_image_url.startswith('/'):
                            relative_image_url = relative_image_url[1:]
                    
                    # For Midjourney
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": model,
                        "promptObject": {
                            "imageUrl": relative_image_url,
                            "mode": mode or "relax",
                            "n": 4,
                            "isNiji6": False,
                            "aspect_width": aspect_width or 1,
                            "aspect_height": aspect_height or 1,
                            "maintainModeration": True
                        }
                    }
                elif model == "dall-e-2":
                    # For Dall-E 2
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "dall-e-2",
                        "promptObject": {
                            "imageUrl": image_url if image_url else image_location,
                            "n": 1,
                            "size": "1024x1024"
                        }
                    }
                elif model == "clipdrop":
                    # For Clipdrop (Stable Diffusion)
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "clipdrop",
                        "promptObject": {
                            "imageUrl": image_url if image_url else image_location
                        }
                    }
                else:
                    # For all other models, we use minimal parameters
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": model,
                        "promptObject": {
                            "imageUrl": image_url if image_url else image_location,
                            "n": int(n)
                        }
                    }

                # Remove the initial slash in Imageurl if it is
                if "imageUrl" in payload["promptObject"] and payload["promptObject"]["imageUrl"] and isinstance(
                        payload["promptObject"]["imageUrl"], str) and payload["promptObject"]["imageUrl"].startswith(
                    '/'):
                    payload["promptObject"]["imageUrl"] = payload["promptObject"]["imageUrl"][1:]
                    logger.debug(
                        f"[{request_id}] Removed leading slash from imageUrl: {payload['promptObject']['imageUrl']}")

                # For VIP users, add Credit to the request
                if api_key.startswith("vip-"):
                    payload["credits"] = 90000  # Standard number of loans for VIP

                # Detailed Payload logistics for debugging
                logger.info(f"[{request_id}] {model} variation payload: {json.dumps(payload, indent=2)}")

                # Using Timeout for all models (10 minutes)
                timeout = MIDJOURNEY_TIMEOUT

                logger.debug(f"[{request_id}] Sending variation request to {ONE_MIN_API_URL}")

                # We send a request to create a variation
                variation_response = api_request(
                    "POST",
                    f"{ONE_MIN_API_URL}",
                    headers={"API-KEY": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout
                )

                if variation_response.status_code != 200:
                    # We process the 504 error for Midjourney in a special way
                    if variation_response.status_code == 504 and model.startswith("midjourney"):
                        logger.error(
                            f"[{request_id}] Received a 504 Gateway Timeout for Midjourney variations. Returning the error to the client.")
                        return (
                            jsonify(
                                {"error": "Gateway Timeout (504) occurred while processing image variation request."}),
                            504,
                        )
                    # For other errors, we continue to try the next model
                    logger.error(
                        f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
                    continue

                # We process the answer and form the result
                variation_data = variation_response.json()
                # Добавляем детальный лог для midjourney модели
                if model.startswith("midjourney"):
                    logger.info(f"[{request_id}] Full Midjourney variation response: {json.dumps(variation_data, indent=2)}")
                logger.debug(f"[{request_id}] Variation response: {variation_data}")

                # We extract the URL variations - initialize an empty array before searching
                variation_urls = []

                # We are trying to find URL variations in the answer - various structures for different models
                if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                    record_detail = variation_data["aiRecord"]["aiRecordDetail"]
                    if "resultObject" in record_detail:
                        result = record_detail["resultObject"]
                        if isinstance(result, list):
                            variation_urls = result
                        elif isinstance(result, str):
                            variation_urls = [result]

                # An alternative search path
                if not variation_urls and "resultObject" in variation_data:
                    result = variation_data["resultObject"]
                    if isinstance(result, list):
                        variation_urls = result
                    elif isinstance(result, str):
                        variation_urls = [result]

                # Search in Data.URL for Dall-E 2
                if not variation_urls and "data" in variation_data and isinstance(variation_data["data"], list):
                    for item in variation_data["data"]:
                        if "url" in item:
                            variation_urls.append(item["url"])

                if not variation_urls:
                    logger.error(f"[{request_id}] No variation URLs found in response with model {model}")
                    continue  # We try the next model

                # Successfully received variations, we leave the cycle
                logger.info(f"[{request_id}] Successfully generated variations with model {model}")
                break

        except Exception as e:
            logger.error(f"[{request_id}] Exception during variation request with model {model}: {str(e)}")
            continue  # We try the next model

    # Clean the temporary file
    try:
        os.unlink(temp_file.name)
    except:
        pass

    # We check if you managed to get variations from any of the models
    if not variation_urls:
        session.close()
        return jsonify({"error": "Failed to create image variations with any available model"}), 500

    # We form complete URL for variations
    full_variation_urls = []
    asset_host = "https://asset.1min.ai"

    for url in variation_urls:
        if not url:
            continue

        # We save the relative path for the API, but create a full URL for display
        relative_url = url
        # If the URL contains a domain, we extract a relative path
        if "asset.1min.ai/" in url:
            relative_url = url.split('asset.1min.ai/', 1)[1]
            # Remove the initial slash if it is
            if relative_url.startswith('/'):
                relative_url = relative_url[1:]
        # If the URL is already without a domain, but starts with the slashus, we remove the slash
        elif url.startswith('/'):
            relative_url = url[1:]

        # Create a full URL to display the user
        if not url.startswith("http"):
            if url.startswith("/"):
                full_url = f"{asset_host}{url}"
            else:
                full_url = f"{asset_host}/{url}"
        else:
            full_url = url

        # We keep the relative path and full URL
        full_variation_urls.append({
            "relative_path": relative_url,
            "full_url": full_url
        })

    # We form an answer in Openai format
    openai_data = []
    for url_data in full_variation_urls:
        # We use the relative path for the API
        openai_data.append({"url": url_data["relative_path"]})

    openai_response = {
        "created": int(time.time()),
        "data": openai_data,
    }

    # Add the text with variation buttons for Markdown Object
    markdown_text = ""
    if len(full_variation_urls) == 1:
        # We use the full URL to display
        markdown_text = f"![Variation]({full_variation_urls[0]['full_url']}) `[_V1_]`"
        # Add a hint to create variations
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** and send it (paste) in the next **prompt**"
    else:
        # We form a text with images and buttons of variations on one line
        image_lines = []

        for i, url_data in enumerate(full_variation_urls):
            # We use the full URL to display
            image_lines.append(f"![Variation {i + 1}]({url_data['full_url']}) `[_V{i + 1}_]`")

        # Combine lines with a new line between them
        markdown_text = "\n".join(image_lines)
        # Add a hint to create variations
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**"

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
    logger.info(
        f"[{request_id}] Successfully generated {len(openai_data)} image variations using model {current_model}")
    return jsonify(openai_response), 200


@app.route("/v1/images/variations", methods=["POST", "OPTIONS"])
@limiter.limit("500 per minute")
def image_variations():
    if request.method == "OPTIONS":
        return handle_options_request()

    # Create a unique ID for request
    request_id = str(uuid.uuid4())
    logger.debug(f"[{request_id}] Processing image variation request")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)
    api_key = auth_header.split(" ")[1]

    # We check whether a request has come with the REQUEST_ID parameter (redirection from/V1/Chat/Complets)
    if 'request_id' in request.args and 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
        # We get data on variation from MemcacheD
        redirect_request_id = request.args.get('request_id')
        variation_key = f"variation:{redirect_request_id}"
        logger.info(f"[{request_id}] Looking for variation data in memcached with key: {variation_key}")
        
        variation_data_json = safe_memcached_operation('get', variation_key)
        
        if variation_data_json:
            logger.info(f"[{request_id}] Found variation data in memcached: {variation_data_json}")
            try:
                if isinstance(variation_data_json, str):
                    variation_data = json.loads(variation_data_json)
                elif isinstance(variation_data_json, bytes):
                    variation_data = json.loads(variation_data_json.decode('utf-8'))
                else:
                    variation_data = variation_data_json

                # We get the way to the temporary file, model and number of variations
                temp_file_path = variation_data.get("temp_file")
                model = variation_data.get("model")
                n = variation_data.get("n", 1)
                # We get a relative path from the data if it was preserved
                image_path = variation_data.get("image_path")

                logger.debug(
                    f"[{request_id}] Retrieved variation data from memcached: model={model}, n={n}, temp_file={temp_file_path}")
                if image_path:
                    logger.debug(f"[{request_id}] Retrieved image path from memcached: {image_path}")

                # We check that the file exists
                file_exists = os.path.exists(temp_file_path)
                logger.info(f"[{request_id}] Temporary file exists: {file_exists}, path: {temp_file_path}")
                
                if file_exists:
                    # We download the file and process directly
                    try:
                        with open(temp_file_path, 'rb') as f:
                            file_data = f.read()
                        
                        file_size = len(file_data)
                        logger.info(f"[{request_id}] Read temporary file, size: {file_size} bytes")

                        # Create a temporary file for processing a request
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        temp_file.write(file_data)
                        temp_file.close()
                        logger.info(f"[{request_id}] Created new temporary file: {temp_file.name}")

                        # Create a file object for the Image_variations route
                        from io import BytesIO
                        file_data_io = BytesIO(file_data)

                        # We register the file in Request.files via Workraund
                        from werkzeug.datastructures import FileStorage
                        file_storage = FileStorage(
                            stream=file_data_io,
                            filename="variation.png",
                            content_type="image/png",
                        )
                        
                        logger.info(f"[{request_id}] Created FileStorage object for image")

                        # We process a request with a new temporary file
                        request.files = {"image": file_storage}
                        logger.info(f"[{request_id}] Added file to request.files")

                        # Create a form with the necessary parameters
                        form_data = [("model", model), ("n", str(n))]

                        # If there is a relative path, add it to the form
                        if image_path:
                            form_data.append(("image_path", image_path))
                            logger.info(f"[{request_id}] Added image_path to form_data: {image_path}")

                        request.form = MultiDict(form_data)
                        logger.info(f"[{request_id}] Set request.form with data: {form_data}")

                        logger.info(f"[{request_id}] Using file from memcached for image variations")

                        # We delete the original temporary file
                        try:
                            os.unlink(temp_file_path)
                            logger.debug(f"[{request_id}] Deleted original temporary file: {temp_file_path}")
                        except Exception as e:
                            logger.warning(f"[{request_id}] Failed to delete original temporary file: {str(e)}")

                        # We will use the original temporary file instead of creating a new
                        # to avoid problems with the closing of the flow
                    except Exception as e:
                        logger.error(f"[{request_id}] Error processing file from memcached: {str(e)}")
                        return jsonify({"error": f"Error processing variation request: {str(e)}"}), 500
                else:
                    logger.error(f"[{request_id}] Temporary file not found: {temp_file_path}")
                    return jsonify({"error": "Image file not found"}), 400
            except Exception as e:
                logger.error(f"[{request_id}] Error processing variation data: {str(e)}")
                return jsonify({"error": f"Error processing variation request: {str(e)}"}), 500
        else:
            logger.error(f"[{request_id}] No variation data found in memcached with key: {variation_key}")
            return jsonify({"error": "No variation data found"}), 400

    # Getting an image file
    if "image" not in request.files:
        logger.error(f"[{request_id}] No image file provided")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    original_model = request.form.get("model", "dall-e-2").strip()
    n = int(request.form.get("n", 1))
    size = request.form.get("size", "1024x1024")
    prompt_text = request.form.get("prompt", "")  # We extract the industrial plane from the request if it is
    mode = request.form.get("mode", "relax")  # We get a regime from a request

    # We check whether the relative path to the image in the Form-data has been transmitted
    relative_image_path = request.form.get("image_path")
    if relative_image_path:
        logger.debug(f"[{request_id}] Using relative image path from form: {relative_image_path}")

    logger.debug(f"[{request_id}] Original model requested: {original_model} for image variations")

    # Determine the order of models for Fallback
    fallback_models = ["midjourney_6_1", "midjourney", "clipdrop", "dall-e-2"]

    # If the requested model supports variations, we try it first
    if original_model in IMAGE_VARIATION_MODELS:
        # We start with the requested model, then we try others, excluding the already requested
        models_to_try = [original_model] + [m for m in fallback_models if m != original_model]
    else:
        # If the requested model does not support variations, we start with Fallback models
        logger.warning(
            f"[{request_id}] Model {original_model} does not support image variations. Will try fallback models")
        models_to_try = fallback_models

    # We save a temporary file for multiple use
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image_file.save(temp_file.name)
        temp_file.close()
    except Exception as e:
        logger.error(f"[{request_id}] Failed to save temporary file: {str(e)}")
        return jsonify({"error": "Failed to process image file"}), 500

    # Create a session to download the image
    session = create_session()
    headers = {"API-KEY": api_key}

    # We extract the ratio of the parties from the industrial plane if it is
    aspect_width = 1
    aspect_height = 1
    if "--ar" in prompt_text:
        ar_match = re.search(r'--ar\s+(\d+):(\d+)', prompt_text)
        if ar_match:
            aspect_width = int(ar_match.group(1))
            aspect_height = int(ar_match.group(2))
            logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_width}:{aspect_height}")

    # Initialize the variable for variations in front of the cycle
    variation_urls = []
    current_model = None

    # We try each model in turn
    for model in models_to_try:
        logger.info(f"[{request_id}] Trying model: {model} for image variations")
        current_model = model

        try:
            # Special processing for Dall-E 2
            if model == "dall-e-2":
                # For Dall-E 2, you need to use a special Openai and direct file transfer
                logger.debug(f"[{request_id}] Special handling for DALL-E 2 variations")

                # Open the image file and create a request
                with open(temp_file.name, 'rb') as img_file:
                    # Openai expects a file directly to Form-Data
                    dalle_files = {
                        'image': (os.path.basename(temp_file.name), img_file, 'image/png')
                    }

                    # Request parameters
                    dalle_form_data = {
                        'n': n,
                        'size': size,
                        'model': 'dall-e-2'
                    }

                    # We create a request for variation directly to Openai API
                    try:
                        # Try to use a direct connection to Openai if available
                        openai_api_key = os.environ.get("OPENAI_API_KEY")
                        if openai_api_key:
                            openai_headers = {"Authorization": f"Bearer {openai_api_key}"}
                            openai_url = "https://api.openai.com/v1/images/variations"

                            logger.debug(f"[{request_id}] Trying direct OpenAI API for DALL-E 2 variations")
                            variation_response = requests.post(
                                openai_url,
                                files=dalle_files,
                                data=dalle_form_data,
                                headers=openai_headers,
                                timeout=300
                            )

                            if variation_response.status_code == 200:
                                logger.debug(f"[{request_id}] OpenAI API variation successful")
                                variation_data = variation_response.json()

                                # We extract the URL from the answer
                                if "data" in variation_data and isinstance(variation_data["data"], list):
                                    for item in variation_data["data"]:
                                        if "url" in item:
                                            variation_urls.append(item["url"])

                                if variation_urls:
                                    logger.info(
                                        f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2 via OpenAI API")
                                    # We form an answer in Openai API format
                                    response_data = {
                                        "created": int(time.time()),
                                        "data": [{"url": url} for url in variation_urls]
                                    }
                                    return jsonify(response_data)
                            else:
                                logger.error(
                                    f"[{request_id}] OpenAI API variation failed: {variation_response.status_code}, {variation_response.text}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error trying direct OpenAI API: {str(e)}")

                    # If the direct request to Openai failed, we try through 1min.ai API
                    try:
                        # We reject the file because it could be read in the previous request
                        img_file.seek(0)

                        # We draw a request through our own and 1min.ai and dall-e 2
                        onemin_url = "https://api.1min.ai/api/features/images/variations"

                        logger.debug(f"[{request_id}] Trying 1min.ai API for DALL-E 2 variations")
                        dalle_onemin_headers = {"API-KEY": api_key}
                        variation_response = requests.post(
                            onemin_url,
                            files=dalle_files,
                            data=dalle_form_data,
                            headers=dalle_onemin_headers,
                            timeout=300
                        )

                        if variation_response.status_code == 200:
                            logger.debug(f"[{request_id}] 1min.ai API variation successful")
                            variation_data = variation_response.json()

                            # We extract the URL from the answer
                            if "data" in variation_data and isinstance(variation_data["data"], list):
                                for item in variation_data["data"]:
                                    if "url" in item:
                                        variation_urls.append(item["url"])

                            if variation_urls:
                                logger.info(
                                    f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2 via 1min.ai API")
                                # We form an answer in Openai API format
                                response_data = {
                                    "created": int(time.time()),
                                    "data": [{"url": url} for url in variation_urls]
                                }
                                return jsonify(response_data)
                        else:
                            logger.error(
                                f"[{request_id}] 1min.ai API variation failed: {variation_response.status_code}, {variation_response.text}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error trying 1min.ai API: {str(e)}")

                # If you could not create a variation with Dall-E 2, we continue with other models
                logger.warning(f"[{request_id}] Failed to create variations with DALL-E 2, trying next model")
                continue

            # For other models, we use standard logic
            # Image loading in 1min.ai
            with open(temp_file.name, 'rb') as img_file:
                files = {"asset": (os.path.basename(temp_file.name), img_file, "image/png")}

                asset_response = session.post(
                    ONE_MIN_ASSET_URL, files=files, headers=headers
                )
                logger.debug(
                    f"[{request_id}] Image upload response status code: {asset_response.status_code}"
                )

                if asset_response.status_code != 200:
                    logger.error(
                        f"[{request_id}] Failed to upload image: {asset_response.status_code} - {asset_response.text}"
                    )
                    continue  # We try the next model

                # Extract an ID of a loaded image and a full URL
                asset_data = asset_response.json()
                logger.debug(f"[{request_id}] Asset upload response: {asset_data}")

                # We get a URL or ID image
                image_id = None
                image_url = None
                image_location = None

                # We are looking for ID in different places of the response structure
                if "id" in asset_data:
                    image_id = asset_data["id"]
                elif "fileContent" in asset_data and "id" in asset_data["fileContent"]:
                    image_id = asset_data["fileContent"]["id"]
                elif "fileContent" in asset_data and "uuid" in asset_data["fileContent"]:
                    image_id = asset_data["fileContent"]["uuid"]

                # We are looking for an absolute URL (location) for image
                if "asset" in asset_data and "location" in asset_data["asset"]:
                    image_location = asset_data["asset"]["location"]
                    # Extract a relative path if the URL contains the domain
                    if image_location and "asset.1min.ai/" in image_location:
                        image_location = image_location.split('asset.1min.ai/', 1)[1]
                    # Remove the initial slash if necessary
                    if image_location and image_location.startswith('/'):
                        image_location = image_location[1:]
                    logger.debug(f"[{request_id}] Using relative path for image location: {image_location}")

                # If there is a Path, we use it as a URL image
                if "fileContent" in asset_data and "path" in asset_data["fileContent"]:
                    image_url = asset_data["fileContent"]["path"]
                    # Extract a relative path if the URL contains the domain
                    if "asset.1min.ai/" in image_url:
                        image_url = image_url.split('asset.1min.ai/', 1)[1]
                    # Remove the initial slash if necessary
                    if image_url.startswith('/'):
                        image_url = image_url[1:]
                    logger.debug(f"[{request_id}] Using relative path for image: {image_url}")

                if not (image_id or image_url or image_location):
                    logger.error(f"[{request_id}] Failed to extract image information from response")
                    continue  # We try the next model

                # We form Payload for image variation
                # We determine which model to use
                if model.startswith("midjourney"):
                    # Проверяем, содержит ли URL домен asset.1min.ai
                    if image_url and "asset.1min.ai/" in image_url:
                        # Извлекаем только относительный путь из URL
                        relative_image_url = image_url.split('asset.1min.ai/', 1)[1]
                        # Удаляем начальный слеш, если он есть
                        if relative_image_url.startswith('/'):
                            relative_image_url = relative_image_url[1:]
                        logger.info(f"[{request_id}] Extracted relative URL for Midjourney: {relative_image_url}")
                    else:
                        relative_image_url = image_url if image_url else image_location
                        if relative_image_url and relative_image_url.startswith('/'):
                            relative_image_url = relative_image_url[1:]
                    
                    # For Midjourney
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": model,
                        "promptObject": {
                            "imageUrl": relative_image_url,
                            "mode": mode or "relax",
                            "n": 4,
                            "isNiji6": False,
                            "aspect_width": aspect_width or 1,
                            "aspect_height": aspect_height or 1,
                            "maintainModeration": True
                        }
                    }
                elif model == "dall-e-2":
                    # For Dall-E 2
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "dall-e-2",
                        "promptObject": {
                            "imageUrl": image_url if image_url else image_location,
                            "n": 1,
                            "size": "1024x1024"
                        }
                    }
                elif model == "clipdrop":
                    # For Clipdrop (Stable Diffusion)
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "clipdrop",
                        "promptObject": {
                            "imageUrl": image_url if image_url else image_location
                        }
                    }
                else:
                    # For all other models, we use minimal parameters
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": model,
                        "promptObject": {
                            "imageUrl": image_url if image_url else image_location,
                            "n": int(n)
                        }
                    }

                # Remove the initial slash in Imageurl if it is
                if "imageUrl" in payload["promptObject"] and payload["promptObject"]["imageUrl"] and isinstance(
                        payload["promptObject"]["imageUrl"], str) and payload["promptObject"]["imageUrl"].startswith(
                    '/'):
                    payload["promptObject"]["imageUrl"] = payload["promptObject"]["imageUrl"][1:]
                    logger.debug(
                        f"[{request_id}] Removed leading slash from imageUrl: {payload['promptObject']['imageUrl']}")

                # For VIP users, add Credit to the request
                if api_key.startswith("vip-"):
                    payload["credits"] = 90000  # Standard number of loans for VIP

                # Detailed Payload logistics for debugging
                logger.info(f"[{request_id}] {model} variation payload: {json.dumps(payload, indent=2)}")

                # Using Timeout for all models (10 minutes)
                timeout = MIDJOURNEY_TIMEOUT

                logger.debug(f"[{request_id}] Sending variation request to {ONE_MIN_API_URL}")

                # We send a request to create a variation
                variation_response = api_request(
                    "POST",
                    f"{ONE_MIN_API_URL}",
                    headers={"API-KEY": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout
                )

                if variation_response.status_code != 200:
                    # We process the 504 error for Midjourney in a special way
                    if variation_response.status_code == 504 and model.startswith("midjourney"):
                        logger.error(
                            f"[{request_id}] Received a 504 Gateway Timeout for Midjourney variations. Returning the error to the client.")
                        return (
                            jsonify(
                                {"error": "Gateway Timeout (504) occurred while processing image variation request."}),
                            504,
                        )
                    # For other errors, we continue to try the next model
                    logger.error(
                        f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
                    continue

                # We process the answer and form the result
                variation_data = variation_response.json()
                # Добавляем детальный лог для midjourney модели
                if model.startswith("midjourney"):
                    logger.info(f"[{request_id}] Full Midjourney variation response: {json.dumps(variation_data, indent=2)}")
                logger.debug(f"[{request_id}] Variation response: {variation_data}")

                # We extract the URL variations - initialize an empty array before searching
                variation_urls = []

                # We are trying to find URL variations in the answer - various structures for different models
                if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                    record_detail = variation_data["aiRecord"]["aiRecordDetail"]
                    if "resultObject" in record_detail:
                        result = record_detail["resultObject"]
                        if isinstance(result, list):
                            variation_urls = result
                        elif isinstance(result, str):
                            variation_urls = [result]

                # An alternative search path
                if not variation_urls and "resultObject" in variation_data:
                    result = variation_data["resultObject"]
                    if isinstance(result, list):
                        variation_urls = result
                    elif isinstance(result, str):
                        variation_urls = [result]

                # Search in Data.URL for Dall-E 2
                if not variation_urls and "data" in variation_data and isinstance(variation_data["data"], list):
                    for item in variation_data["data"]:
                        if "url" in item:
                            variation_urls.append(item["url"])

                if not variation_urls:
                    logger.error(f"[{request_id}] No variation URLs found in response with model {model}")
                    continue  # We try the next model

                # Successfully received variations, we leave the cycle
                logger.info(f"[{request_id}] Successfully generated variations with model {model}")
                break

        except Exception as e:
            logger.error(f"[{request_id}] Exception during variation request with model {model}: {str(e)}")
            continue  # We try the next model

    # Clean the temporary file
    try:
        os.unlink(temp_file.name)
    except:
        pass

    # We check if you managed to get variations from any of the models
    if not variation_urls:
        session.close()
        return jsonify({"error": "Failed to create image variations with any available model"}), 500

    # We form complete URL for variations
    full_variation_urls = []
    asset_host = "https://asset.1min.ai"

    for url in variation_urls:
        if not url:
            continue

        # We save the relative path for the API, but create a full URL for display
        relative_url = url
        # If the URL contains a domain, we extract a relative path
        if "asset.1min.ai/" in url:
            relative_url = url.split('asset.1min.ai/', 1)[1]
            # Remove the initial slash if it is
            if relative_url.startswith('/'):
                relative_url = relative_url[1:]
        # If the URL is already without a domain, but starts with the slashus, we remove the slash
        elif url.startswith('/'):
            relative_url = url[1:]

        # Create a full URL to display the user
        if not url.startswith("http"):
            if url.startswith("/"):
                full_url = f"{asset_host}{url}"
            else:
                full_url = f"{asset_host}/{url}"
        else:
            full_url = url

        # We keep the relative path and full URL
        full_variation_urls.append({
            "relative_path": relative_url,
            "full_url": full_url
        })

    # We form an answer in Openai format
    openai_data = []
    for url_data in full_variation_urls:
        # We use the relative path for the API
        openai_data.append({"url": url_data["relative_path"]})

    openai_response = {
        "created": int(time.time()),
        "data": openai_data,
    }

    # Add the text with variation buttons for Markdown Object
    markdown_text = ""
    if len(full_variation_urls) == 1:
        # We use the full URL to display
        markdown_text = f"![Variation]({full_variation_urls[0]['full_url']}) `[_V1_]`"
        # Add a hint to create variations
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** and send it (paste) in the next **prompt**"
    else:
        # We form a text with images and buttons of variations on one line
        image_lines = []

        for i, url_data in enumerate(full_variation_urls):
            # We use the full URL to display
            image_lines.append(f"![Variation {i + 1}]({url_data['full_url']}) `[_V{i + 1}_]`")

        # Combine lines with a new line between them
        markdown_text = "\n".join(image_lines)
        # Add a hint to create variations
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**"

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
    logger.info(
        f"[{request_id}] Successfully generated {len(openai_data)} image variations using model {current_model}")
    return jsonify(openai_response), 200


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

    # Creating a conversation with PDF in 1min.ai
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
        # Output of the response structure for debugging
        logger.debug(f"Response structure: {json.dumps(one_min_response)[:200]}...")

        # We get an answer from the appropriate place to json
        result_text = (
            one_min_response.get("aiRecord", {})
            .get("aiRecordDetail", {})
            .get("resultObject", [""])[0]
        )

        if not result_text:
            # Alternative ways to extract an answer
            if "resultObject" in one_min_response:
                result_text = (
                    one_min_response["resultObject"][0]
                    if isinstance(one_min_response["resultObject"], list)
                    else one_min_response["resultObject"]
                )
            elif "result" in one_min_response:
                result_text = one_min_response["result"]
            else:
                # If you have not found an answer along the well -known paths, we return the error
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
        # Return the error in the format compatible with Openai
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
    response.headers["Access-Control-Allow-Origin"] = "*"  # Corrected the hyphen in the title name
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    # Add more Cors headings
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
    return response  # Return the answer for the chain


def stream_response(response, request_data, model, prompt_tokens, session=None):
    """
    Stream received from 1min.ai response in a format compatible with Openai API.
    """
    all_chunks = ""

    # We send the first fragment: the role of the message
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

    # Simple implementation for content processing
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

    # Final cup denoting the end of the flow
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
    Safely creates a temporary file and guarantees its deletion after use

    Args:
        Prefix: Prefix for file name
        Request_id: ID Request for Logging

    Returns:
        STR: Way to the temporary file
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

    # Create a temporary directory if it is not
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Clean old files (over 1 hour)
    try:
        current_time = time.time()
        for old_file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, old_file)
            if os.path.isfile(file_path):
                # If the file is older than 1 hour - delete
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

    # Create a new temporary file
    temp_file_path = os.path.join(temp_dir, f"{prefix}_{request_id}_{random_string}")
    return temp_file_path


def retry_image_upload(image_url, api_key, request_id=None):
    """Uploads an image with repeated attempts, returns a direct link to it"""
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Uploading image: {image_url}")

    # We create a new session for this request
    session = create_session()
    temp_file_path = None

    try:
        # We load the image
        if image_url.startswith(("http://", "https://")):
            # URL loading
            logger.debug(f"[{request_id}] Fetching image from URL: {image_url}")
            response = session.get(image_url, stream=True)
            response.raise_for_status()
            image_data = response.content
        else:
            # Decoding Base64
            logger.debug(f"[{request_id}] Decoding base64 image")
            image_data = base64.b64decode(image_url.split(",")[1])

        # Check the file size
        if len(image_data) == 0:
            logger.error(f"[{request_id}] Empty image data")
            return None

        # Create a temporary file
        temp_file_path = safe_temp_file("image", request_id)

        with open(temp_file_path, "wb") as f:
            f.write(image_data)

        # Check that the file is not empty
        if os.path.getsize(temp_file_path) == 0:
            logger.error(f"[{request_id}] Empty image file created: {temp_file_path}")
            return None

        # We load to the server
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

                # We get URL images
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

                # We get the path to the file from FileContent
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
        # Close the session
        session.close()
        # We delete a temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"[{request_id}] Removed temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(
                    f"[{request_id}] Failed to remove temp file {temp_file_path}: {str(e)}"
                )


def create_session():
    """Creates a new session with optimal settings for APIs"""
    session = requests.Session()

    # Setting up repeated attempts for all requests
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
    Downloads the file/document to the server and returns its ID.

    Args:
        File_DATA: Binar file contents
        File_name: file name
        API_KEY: user API
        Request_id: ID Request for Logging

    Returns:
        STR: ID loaded file or None in case of error
    """
    session = create_session()
    try:
        # Determine the type of expansion file
        extension = os.path.splitext(file_name)[1].lower()
        logger.info(f"[{request_id}] Uploading document: {file_name}")

        # Dictionary with MIME types for different file extensions
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

        # We get MIME-type from a dictionary or use Octet-Stream by default
        mime_type = mime_types.get(extension, "application/octet-stream")

        # Determine the type of file for special processing
        file_type = None
        if extension in [".doc"]:
            file_type = "DOC"
        elif extension in [".docx"]:
            file_type = "DOCX"

        # We download the file to the server - add more details to the logs
        logger.info(
            f"[{request_id}] Uploading file to 1min.ai: {file_name} ({mime_type}, {len(file_data)} bytes)"
        )

        headers = {"API-KEY": api_key}

        # Special headlines for DOC/DOCX
        if file_type in ["DOC", "DOCX"]:
            headers["X-File-Type"] = file_type

        files = {"asset": (file_name, file_data, mime_type)}

        upload_response = session.post(ONE_MIN_ASSET_URL, headers=headers, files=files)

        if upload_response.status_code != 200:
            logger.error(
                f"[{request_id}] Document upload failed: {upload_response.status_code} - {upload_response.text}"
            )
            return None

        # Detailed logistics of the answer
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
                # We are trying to find ID in other places of response structure
                if isinstance(response_data, dict):
                    # Recursive search for ID in the response structure
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
    File download route (analogue Openai Files API)
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
        # We save the file in memory
        file_data = file.read()
        file_name = file.filename

        # We download the file to the 1min.ai server
        file_id = upload_document(file_data, file_name, api_key, request_id)

        if not file_id:
            return jsonify({"error": "Failed to upload file"}), 500

        # We save the file of the file in the user's session through Memcache, if it is available
        if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
            try:
                user_key = f"user:{api_key}"
                # We get the current user's current files or create a new list
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

                # Add a new file
                file_info = {
                    "id": file_id,
                    "filename": file_name,
                    "uploaded_at": int(time.time())
                }

                # Check that a file with such an ID is not yet on the list
                if not any(f.get("id") == file_id for f in user_files):
                    user_files.append(file_info)

                # We save the updated file list
                safe_memcached_operation('set', user_key, json.dumps(user_files))
                logger.info(f"[{request_id}] Saved file ID {file_id} for user in memcached")

                # Add the user to the list of well -known users for cleaning function
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

                # Add the API key to the list of famous users if it is not yet
                if api_key not in known_users_list:
                    known_users_list.append(api_key)
                    safe_memcached_operation('set', 'known_users_list', json.dumps(known_users_list))
                    logger.debug(f"[{request_id}] Added user to known_users_list for cleanup")
            except Exception as e:
                logger.error(f"[{request_id}] Error saving file info to memcached: {str(e)}")

        # We create an answer in the Openai API format
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
    Emulates a streaming response for cases when the API does not support the flow gear

    Args:
        Full_Content: Full text of the answer
        Request_Data: Request data
        Model: Model
        Prompt_tokens: the number of tokens in the request

    Yields:
        STR: Lines for streaming
    """
    # We break the answer to fragments by ~ 5 words
    words = full_content.split()
    chunks = [" ".join(words[i: i + 5]) for i in range(0, len(words), 5)]

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
        time.sleep(0.05)  # Small delay in emulating stream

    # We calculate the tokens
    tokens = calculate_token(full_content)

    # Final chambers
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


# A function for performing a request to the API with a new session
def api_request(req_method, url, headers=None,
                requester_ip=None, data=None,
                files=None, stream=False,
                timeout=None, json=None, **kwargs):
    """Performs the HTTP request to the API with the normalization of the URL and error processing"""
    req_url = url.strip()
    logger.debug(f"API request URL: {req_url}")

    # Request parameters
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

    # Add other parameters
    req_params.update(kwargs)

    # We check whether the request is an operation with images
    is_image_operation = False
    if json and isinstance(json, dict):
        operation_type = json.get("type", "")
        if operation_type in [IMAGE_GENERATOR, IMAGE_VARIATOR]:
            is_image_operation = True
            logger.debug(f"Detected image operation: {operation_type}, using extended timeout")

    # We use increased timaut for operations with images
    if is_image_operation:
        req_params["timeout"] = timeout or MIDJOURNEY_TIMEOUT
        logger.debug(f"Using extended timeout for image operation: {MIDJOURNEY_TIMEOUT}s")
    else:
        req_params["timeout"] = timeout or DEFAULT_TIMEOUT

    # We fulfill the request
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
    Route for converting speech into text (analogue of Openai Whisper API)
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

    # Checking the availability of the Audio file
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
        # We create a new session for loading audio
        session = create_session()
        headers = {"API-KEY": api_key}

        # Audio loading in 1min.ai
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

            # We form Payload for request Speech_to_text
            payload = {
                "type": "SPEECH_TO_TEXT",
                "model": "whisper-1",
                "promptObject": {
                    "audioUrl": audio_path,
                    "response_format": response_format,
                },
            }

        # Add additional parameters if they are provided
        if language:
            payload["promptObject"]["language"] = language

        if temperature is not None:
            payload["promptObject"]["temperature"] = float(temperature)

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}

        # We send a request
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

        # We convert the answer to the Openai API format
        one_min_response = response.json()

        try:
            # We extract the text from the answer
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

            # Check if the result_text json is
            try:
                # If result_text is a json line, we rush it
                if result_text and result_text.strip().startswith("{"):
                    parsed_json = json.loads(result_text)
                    # If Parsed_json has a "Text" field, we use its value
                    if "text" in parsed_json:
                        result_text = parsed_json["text"]
                        logger.debug(f"[{request_id}] Extracted inner text from JSON: {result_text}")
            except (json.JSONDecodeError, TypeError, ValueError):
                # If it was not possible to steam like JSON, we use it as it is
                logger.debug(f"[{request_id}] Using result_text as is: {result_text}")
                pass

            if not result_text:
                logger.error(
                    f"[{request_id}] Could not extract transcription text from API response"
                )
                return jsonify({"error": "Could not extract transcription text"}), 500

            # The most simple and reliable response format
            logger.info(f"[{request_id}] Successfully processed audio transcription: {result_text}")

            # Create json strictly in Openai API format
            response_data = {"text": result_text}

            # Add Cors headlines
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
    Route for translating audio to text (analogue Openai Whisper API)
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

    # Checking the availability of the Audio file
    if "file" not in request.files:
        logger.error(f"[{request_id}] No audio file provided")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["file"]
    model = request.form.get("model", "whisper-1")
    response_format = request.form.get("response_format", "text")
    temperature = request.form.get("temperature", 0)

    logger.info(f"[{request_id}] Processing audio translation with model {model}")

    try:
        # We create a new session for loading audio
        session = create_session()
        headers = {"API-KEY": api_key}

        # Audio loading in 1min.ai
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

            # We form Payload for request Audio_Translator
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

        # We send a request
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

        # We convert the answer to the Openai API format
        one_min_response = response.json()

        try:
            # We extract the text from the answer
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

            # The most simple and reliable response format
            logger.info(f"[{request_id}] Successfully processed audio translation: {result_text}")

            # Create json strictly in Openai API format
            response_data = {"text": result_text}

            # Add Cors headlines
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
    Route for converting text into speech (analogue Openai TTS API)
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = request.args.get('request_id', str(uuid.uuid4())[:8])
    logger.info(f"[{request_id}] Received request: /v1/audio/speech")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # We get data data
    request_data = {}
    
    # We check the availability of data in Memcached if the request has been redirected
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
                    
                # We delete data from the cache, they are no longer needed
                safe_memcached_operation('delete', tts_session_key)
                logger.debug(f"[{request_id}] Retrieved TTS request data from memcached")
        except Exception as e:
            logger.error(f"[{request_id}] Error retrieving TTS session data: {str(e)}")
    
    # If the data is not found in Memcache, we try to get them from the query body
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
        # We form Payload for request_to_Speech
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

        # We send a request
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

        # We process the answer
        one_min_response = response.json()

        try:
            # We get a URL audio from the answer
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

            # We get audio data by URL
            audio_response = api_request("GET", f"https://asset.1min.ai/{audio_url}")

            if audio_response.status_code != 200:
                logger.error(f"[{request_id}] Failed to download audio: {audio_response.status_code}")
                return jsonify({"error": "Failed to download audio"}), 500

            # We return the audio to the client
            logger.info(f"[{request_id}] Successfully generated speech audio")

            # We create an answer with audio and correct MIME-type
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


# Functions for working with files in API
@app.route("/v1/files", methods=["GET", "POST", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_files():
    """
    Route for working with files: getting a list and downloading new files
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Get - getting a list of files
    if request.method == "GET":
        logger.info(f"[{request_id}] Received request: GET /v1/files")
        try:
            # We get a list of user files from MemcacheD
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

                            # Let's convert files about files to API response format
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

            # We form an answer in Openai API format
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

    # Post - downloading a new file
    elif request.method == "POST":
        logger.info(f"[{request_id}] Received request: POST /v1/files")

        # Checking a file
        if "file" not in request.files:
            logger.error(f"[{request_id}] No file provided")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        purpose = request.form.get("purpose", "assistants")

        try:
            # We get the contents of the file
            file_data = file.read()
            file_name = file.filename

            # We get a loaded file ID
            file_id = upload_document(file_data, file_name, api_key, request_id)

            if not file_id:
                logger.error(f"[{request_id}] Failed to upload file")
                return jsonify({"error": "Failed to upload file"}), 500

            # We form an answer in Openai API format
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
    Route for working with a specific file: obtaining information and deleting
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Get - obtaining file information
    if request.method == "GET":
        logger.info(f"[{request_id}] Received request: GET /v1/files/{file_id}")
        try:
            # We are looking for a file in saved user files in Memcache
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

                            # Looking for a file with the specified ID
                            for file_item in user_files:
                                if file_item.get("id") == file_id:
                                    file_info = file_item
                                    logger.debug(f"[{request_id}] Found file info in memcached: {file_id}")
                                    break
                        except Exception as e:
                            logger.error(f"[{request_id}] Error parsing user files from memcached: {str(e)}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")

            # If the file is not found, we return the filler
            if not file_info:
                logger.debug(f"[{request_id}] File not found in memcached, using placeholder: {file_id}")
                file_info = {
                    "id": file_id,
                    "bytes": 0,
                    "created_at": int(time.time()),
                    "filename": f"file_{file_id}"
                }

            # We form an answer in Openai API format
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

    # Delete - File deletion
    elif request.method == "DELETE":
        logger.info(f"[{request_id}] Received request: DELETE /v1/files/{file_id}")
        try:
            # If the files are stored in Memcached, delete the file from the list
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

                            # We filter the list, excluding the file with the specified ID
                            new_user_files = [f for f in user_files if f.get("id") != file_id]

                            # If the list has changed, we save the updated list
                            if len(new_user_files) < len(user_files):
                                safe_memcached_operation('set', user_key, json.dumps(new_user_files))
                                logger.info(f"[{request_id}] Deleted file {file_id} from user's files in memcached")
                                deleted = True
                        except Exception as e:
                            logger.error(f"[{request_id}] Error updating user files in memcached: {str(e)}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")

            # Return the answer about successful removal (even if the file was not found)
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
    Route for obtaining the contents of the file
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
        # In 1min.ai there is no API to obtain the contents of the file by ID
        # Return the error
        logger.error(f"[{request_id}] File content retrieval not supported")
        return jsonify({"error": "File content retrieval not supported"}), 501

    except Exception as e:
        logger.error(f"[{request_id}] Exception during file content request: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Closter function for safe access to Memcache
def safe_memcached_operation(operation, *args, **kwargs):
    """
    Safely performs the operation with Memcache, processing possible errors.

    Args:
        Operation: The name of the Memcached method to execute
        *args, ** kwargs: arguments for the method

    Returns:
        Operation result or None in case of error
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
    Function for periodic deleting all user files
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting scheduled files cleanup task")

    try:
        # We get all users with files from MemcacheD
        if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
            # We get all the keys that begin with "user:"
            try:
                keys = []

                # Instead of scanning Slabs, we use a list of famous users
                # which should be saved when uploading files
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

                # We delete files for each user
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

                        # We delete each file
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
                                        logger.error(
                                            f"[{request_id}] Scheduled cleanup: failed to delete file {file_id}: {delete_response.status_code}")
                                except Exception as e:
                                    logger.error(
                                        f"[{request_id}] Scheduled cleanup: error deleting file {file_id}: {str(e)}")

                        # Cleaning the list of user files
                        safe_memcached_operation('set', user_key, json.dumps([]))
                        logger.info(f"[{request_id}] Cleared files list for user {api_key[:8]}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error processing user {user_key}: {str(e)}")
            except Exception as e:
                logger.error(f"[{request_id}] Error getting keys from memcached: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Error in scheduled cleanup task: {str(e)}")

    # Plan the following execution in an hour
    cleanup_timer = threading.Timer(3600, delete_all_files_task)
    cleanup_timer.daemon = True
    cleanup_timer.start()
    logger.info(f"[{request_id}] Scheduled next cleanup in 1 hour")


def split_text_for_streaming(text, chunk_size=6):
    """
    It breaks the text into small parts to emulate streaming output.

    Args:
        Text (str): text for breakdown
        chunk_size (int): the approximate size of the parts in words

    Returns:
        List: List of parts of the text
    """
    if not text:
        return [""]

    # We break the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # We are grouping sentences to champs
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words_in_sentence = len(sentence.split())

        # If the current cup is empty or the addition of a sentence does not exceed the limit of words
        if not current_chunk or current_word_count + words_in_sentence <= chunk_size:
            current_chunk.append(sentence)
            current_word_count += words_in_sentence
        else:
            # We form a cup and begin the new
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words_in_sentence

    # Add the last cup if it is not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # If there is no Cankov (breakdown did not work), we return the entire text entirely
    if not chunks:
        return [text]

    return chunks


def create_image_variations(image_url, user_model, n, aspect_width=None, aspect_height=None, mode=None,
                            request_id=None):
    """
    Creates variations based on the original image, taking into account the specifics of each model.
    """
    # Initialize the URL list in front of the cycle
    variation_urls = []
    current_model = None

    # We use a temporary ID request if it was not provided
    if request_id is None:
        request_id = str(uuid.uuid4())

    # We get saved generation parameters
    generation_params = None
    if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
        try:
            gen_key = f"gen_params:{request_id}"
            params_json = safe_memcached_operation('get', gen_key)
            if params_json:
                if isinstance(params_json, str):
                    generation_params = json.loads(params_json)
                elif isinstance(params_json, bytes):
                    generation_params = json.loads(params_json.decode('utf-8'))
                logger.debug(f"[{request_id}] Retrieved generation parameters from memcached: {generation_params}")
        except Exception as e:
            logger.error(f"[{request_id}] Error retrieving generation parameters: {str(e)}")

    # We use saved parameters if they are available
    if generation_params:
        # We take Aspect_width and Aspect_Height from saved parameters if they are
        if "aspect_width" in generation_params and "aspect_height" in generation_params:
            aspect_width = generation_params.get("aspect_width")
            aspect_height = generation_params.get("aspect_height")
            logger.debug(f"[{request_id}] Using saved aspect ratio: {aspect_width}:{aspect_height}")

        # We take the mode of saved parameters if it is
        if "mode" in generation_params:
            mode = generation_params.get("mode")
            logger.debug(f"[{request_id}] Using saved mode: {mode}")

    # We determine the list of models for variations
    variation_models = []
    if user_model in VARIATION_SUPPORTED_MODELS:
        variation_models.append(user_model)
    variation_models.extend([m for m in ["midjourney_6_1", "midjourney", "clipdrop", "dall-e-2"] if m != user_model])
    variation_models = list(dict.fromkeys(variation_models))

    logger.info(f"[{request_id}] Trying image variations with models: {variation_models}")

    # Create a session to download the image
    session = create_session()

    try:
        # We load the image
        image_response = session.get(image_url, stream=True, timeout=60)
        if image_response.status_code != 200:
            logger.error(f"[{request_id}] Failed to download image: {image_response.status_code}")
            return jsonify({"error": "Failed to download image"}), 500

        # We try each model in turn
        for model in variation_models:
            current_model = model
            logger.info(f"[{request_id}] Trying model: {model} for image variations")

            try:
                # Determine the MIME-type image based on the contents or url
                content_type = "image/png"  # By default
                if "content-type" in image_response.headers:
                    content_type = image_response.headers["content-type"]
                elif image_url.lower().endswith(".webp"):
                    content_type = "image/webp"
                elif image_url.lower().endswith(".jpg") or image_url.lower().endswith(".jpeg"):
                    content_type = "image/jpeg"
                elif image_url.lower().endswith(".gif"):
                    content_type = "image/gif"

                # Determine the appropriate extension for the file
                ext = "png"
                if "webp" in content_type:
                    ext = "webp"
                elif "jpeg" in content_type or "jpg" in content_type:
                    ext = "jpg"
                elif "gif" in content_type:
                    ext = "gif"

                logger.debug(f"[{request_id}] Detected image type: {content_type}, extension: {ext}")

                # We load the image to the server with the correct MIME type
                files = {"asset": (f"variation.{ext}", image_response.content, content_type)}
                upload_response = session.post(ONE_MIN_ASSET_URL, files=files, headers=headers)

                if upload_response.status_code != 200:
                    logger.error(f"[{request_id}] Image upload failed: {upload_response.status_code}")
                    continue

                upload_data = upload_response.json()
                logger.debug(f"[{request_id}] Asset upload response: {upload_data}")

                # We get the way to the loaded image
                image_path = None
                if "fileContent" in upload_data and "path" in upload_data["fileContent"]:
                    image_path = upload_data["fileContent"]["path"]
                    # We remove the initial slash if it is
                    if image_path.startswith('/'):
                        image_path = image_path[1:]
                    logger.debug(f"[{request_id}] Using relative path for variation: {image_path}")
                else:
                    logger.error(f"[{request_id}] Could not extract image path from upload response")
                    continue

                # We form Payload depending on the model
                if model in ["midjourney_6_1", "midjourney"]:
                    # For Midjourney
                    # Add the URL transformation
                    if image_url and isinstance(image_url, str) and 'asset.1min.ai/' in image_url:
                        image_url = image_url.split('asset.1min.ai/', 1)[1]
                        logger.debug(f"[{request_id}] Extracted path from image_url: {image_url}")

                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": model,
                        "promptObject": {
                            "imageUrl": image_url if image_url else image_location,
                            "mode": mode or request_data.get("mode", "relax"),  # We use the mode from the industrial
                            "n": 4,
                            "isNiji6": False,
                            "aspect_width": aspect_width or 1,
                            "aspect_height": aspect_height or 1,
                            "maintainModeration": True
                        }
                    }
                    # Detailed logistics for Midjourney
                    logger.info(f"[{request_id}] Midjourney variation payload:")
                    logger.info(f"[{request_id}] promptObject: {json.dumps(payload['promptObject'], indent=2)}")
                elif model == "dall-e-2":
                    # For Dall-E 2
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "dall-e-2",
                        "promptObject": {
                            "imageUrl": image_path,
                            "n": 1,
                            "size": "1024x1024"
                        }
                    }
                    logger.info(f"[{request_id}] DALL-E 2 variation payload: {json.dumps(payload, indent=2)}")

                    # We send a request through the main API URL
                    variation_response = api_request(
                        "POST",
                        ONE_MIN_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=300
                    )

                    if variation_response.status_code != 200:
                        logger.error(
                            f"[{request_id}] DALL-E 2 variation failed: {variation_response.status_code}, {variation_response.text}")
                        continue

                    # We process the answer
                    variation_data = variation_response.json()

                    # We extract the URL from the answer
                    if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                        result_object = variation_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
                        if isinstance(result_object, list):
                            variation_urls.extend(result_object)
                        elif isinstance(result_object, str):
                            variation_urls.append(result_object)
                    elif "resultObject" in variation_data:
                        result_object = variation_data["resultObject"]
                        if isinstance(result_object, list):
                            variation_urls.extend(result_object)
                        elif isinstance(result_object, str):
                            variation_urls.append(result_object)

                    if variation_urls:
                        logger.info(
                            f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2")
                        break
                    else:
                        logger.warning(f"[{request_id}] No variation URLs found in DALL-E 2 response")
                elif model == "clipdrop":
                    # For Clipdrop
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "clipdrop",
                        "promptObject": {
                            "imageUrl": image_path,
                            "n": n
                        }
                    }
                    logger.info(f"[{request_id}] Clipdrop variation payload: {json.dumps(payload, indent=2)}")

                    # We send a request through the main API URL
                    variation_response = api_request(
                        "POST",
                        ONE_MIN_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=300
                    )

                    if variation_response.status_code != 200:
                        logger.error(
                            f"[{request_id}] Clipdrop variation failed: {variation_response.status_code}, {variation_response.text}")
                        continue

                    # We process the answer
                    variation_data = variation_response.json()

                    # We extract the URL from the answer
                    if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                        result_object = variation_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
                        if isinstance(result_object, list):
                            variation_urls.extend(result_object)
                        elif isinstance(result_object, str):
                            variation_urls.append(result_object)
                    elif "resultObject" in variation_data:
                        result_object = variation_data["resultObject"]
                        if isinstance(result_object, list):
                            variation_urls.extend(result_object)
                        elif isinstance(result_object, str):
                            variation_urls.append(result_object)

                    if variation_urls:
                        logger.info(
                            f"[{request_id}] Successfully created {len(variation_urls)} variations with Clipdrop")
                        break
                    else:
                        logger.warning(f"[{request_id}] No variation URLs found in Clipdrop response")

                logger.debug(f"[{request_id}] Sending variation request to URL: {ONE_MIN_API_URL}")
                logger.debug(f"[{request_id}] Using headers: {json.dumps(headers)}")

                # We send a request to create a variation
                timeout = MIDJOURNEY_TIMEOUT if model.startswith("midjourney") else DEFAULT_TIMEOUT
                logger.debug(f"Using extended timeout for Midjourney: {timeout}s")

                variation_response = api_request(
                    "POST",
                    ONE_MIN_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )

                if variation_response.status_code != 200:
                    logger.error(
                        f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
                    continue

                # We process the answer
                variation_data = variation_response.json()

                # We extract the URL variations from the answer
                if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                    result_object = variation_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
                    if isinstance(result_object, list):
                        variation_urls.extend(result_object)
                    elif isinstance(result_object, str):
                        variation_urls.append(result_object)
                elif "resultObject" in variation_data:
                    result_object = variation_data["resultObject"]
                    if isinstance(result_object, list):
                        variation_urls.extend(result_object)
                    elif isinstance(result_object, str):
                        variation_urls.append(result_object)

                if variation_urls:
                    logger.info(f"[{request_id}] Successfully created {len(variation_urls)} variations with {model}")
                    break
                else:
                    logger.warning(f"[{request_id}] No variation URLs found in response for model {model}")

            except Exception as e:
                logger.error(f"[{request_id}] Error with model {model}: {str(e)}")
                continue

        # If you could not create variations with any model
        if not variation_urls:
            logger.error(f"[{request_id}] Failed to create variations with any available model")
            return jsonify({"error": "Failed to create image variations with any available model"}), 500

        # We form an answer
        openai_response = {
            "created": int(time.time()),
            "data": []
        }

        for url in variation_urls:
            openai_data = {
                "url": url
            }
            openai_response["data"].append(openai_data)

        # We form a markdown text with a hint
        text_lines = []
        for i, url in enumerate(variation_urls, 1):
            text_lines.append(f"Image {i} ({url}) [_V{i}_]")
        text_lines.append(
            "\n> To generate **variants** of **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**")

        text_response = "\n".join(text_lines)

        openai_response["choices"] = [{
            "message": {
                "role": "assistant",
                "content": text_response
            },
            "index": 0,
            "finish_reason": "stop"
        }]

        logger.info(f"[{request_id}] Returning {len(variation_urls)} variation URLs to client")

        response = jsonify(openai_response)
        return set_response_headers(response)

    except Exception as e:
        logger.error(f"[{request_id}] Exception during image variation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()


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

