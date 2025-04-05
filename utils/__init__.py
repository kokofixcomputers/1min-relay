# Импортируем логгер из модуля utils
from utils.logger import logger
# Импортируем Flask app из app.py
from app import app, limiter

# Импортируем всё содержимое модулей
from .text import *
from .images import *
from .audio import *
from .files import *
