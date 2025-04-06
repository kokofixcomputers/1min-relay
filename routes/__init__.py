# routes/__init__.py
# Полностью переписан для использования прямого импорта app и limiter из app.py

# Импортируем логгер
from utils.logger import logger
from utils.imports import *

logger.info("Инициализация маршрутов начата")
