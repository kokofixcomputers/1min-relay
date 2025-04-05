# version 1.0.3 #increment every time you make changes
# 2025-04-05 18:00 #change to actual date and time every time you make changes
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

# Сначала импортируем constants, затем остальные модули
from utils.constants import *
from utils.common import *
from utils.memcached import *
from routes import *

# Varias of the environment

# Параметры порта и другие настройки окружения
PORT = int(os.getenv("PORT", 5001))

# Глобальные переменные и инициализация
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

# Global storage for use when MemcacheD is not available
MEMORY_STORAGE = {}

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
