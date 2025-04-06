# utils/imports.py
# Центральный файл для всех импортов в проекте

# Стандартные библиотеки Python
import base64
import datetime
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

# Библиотеки Flask и зависимости
try:
    from flask import Flask, request, jsonify, make_response, Response, redirect, url_for
    from werkzeug.datastructures import MultiDict
    from waitress import serve
except ImportError:
    print("Flask или его зависимости не установлены. Установите пакеты: flask, waitress")

# Дополнительные библиотеки (опциональные)
try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv не установлен. Установите пакет: python-dotenv")
    load_dotenv = lambda: None  # Пустая функция-заглушка

try:
    import tiktoken
except ImportError:
    print("tiktoken не установлен. Установите пакет: tiktoken")
    
try:
    import printedcolors
except ImportError:
    print("printedcolors не установлен.")
    # Создаем заглушку для printedcolors
    class ColorStub:
        class fg:
            lightcyan = ""
        reset = ""
    printedcolors = type('', (), {'Color': ColorStub})()

try:
    import requests
except ImportError:
    print("requests не установлен. Установите пакет: requests")

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    print("flask-limiter не установлен. Ограничение запросов будет отключено.")
    LIMITER_AVAILABLE = False
    # Создаем заглушку для Limiter
    class MockLimiter:
        def __init__(self, *args, **kwargs):
            pass
        
        def limit(self, limit_value):
            def decorator(f):
                return f
            return decorator
    
    Limiter = MockLimiter
    get_remote_address = lambda: "127.0.0.1"

try:
    from flask_cors import cross_origin
    CORS_AVAILABLE = True
except ImportError:
    print("flask-cors не установлен. CORS будет отключен.")
    CORS_AVAILABLE = False
    # Создаем заглушку для cross_origin
    def cross_origin(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

try:
    import memcache
    from pymemcache.client.base import Client as PyMemcacheClient
    MEMCACHED_AVAILABLE = True
except ImportError:
    print("pymemcache/python-memcache не установлен. Кэширование будет отключено.")
    MEMCACHED_AVAILABLE = False
    memcache = None
    PyMemcacheClient = None

# Загружаем переменные окружения
load_dotenv()

# Отключаем предупреждения от flask_limiter
warnings.filterwarnings(
    "ignore", category=UserWarning, module="flask_limiter.extension"
) 
