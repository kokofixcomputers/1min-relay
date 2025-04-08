# version 1.0.1 #increment every time you make changes
# utils/imports.py
# Центральный файл для всех импортов в проекте

# Стандартные библиотеки Python
import base64
import datetime
import functools
import hashlib
import json
import os
import random
import re
import socket
import string
import sys
import tempfile
import threading
import time
import traceback
import uuid
import warnings

# Подавляем предупреждения от flask_limiter
warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter.extension")

# Загружаем переменные окружения
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Заглушка для load_dotenv
    def load_dotenv(): pass
    load_dotenv()

# Библиотеки Flask и основные зависимости
from flask import Flask, request, jsonify, make_response, Response, redirect, url_for
from werkzeug.datastructures import MultiDict
from waitress import serve
import requests

# Опциональные библиотеки с заглушками
try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import printedcolors
except ImportError:
    # Заглушка для printedcolors
    class ColorStub:
        class fg:
            lightcyan = ""
        reset = ""
    printedcolors = type('', (), {'Color': ColorStub})()

# Библиотеки для ограничения запросов
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    LIMITER_AVAILABLE = False
    # Заглушка для Limiter
    class MockLimiter:
        def __init__(self, *args, **kwargs): pass
        def limit(self, limit_value):
            def decorator(f): return f
            return decorator
    Limiter = MockLimiter
    get_remote_address = lambda: "127.0.0.1"

# CORS поддержка
try:
    from flask_cors import cross_origin
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    # Заглушка для cross_origin
    def cross_origin(*args, **kwargs):
        def decorator(f): return f
        return decorator

# Библиотеки для работы с Memcached
try:
    import memcache
    from pymemcache.client.base import Client as PyMemcacheClient
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False
    memcache = None
    PyMemcacheClient = None

