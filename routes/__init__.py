# routes/__init__.py
# Полностью переписан для использования прямого импорта app и limiter из app.py

# Импортируем логгер
from utils.logger import logger
from utils.imports import *

# Делаем app и limiter доступными при импорте routes
import sys
mod = sys.modules[__name__]

# Импортируем app и limiter из корневого модуля
try:
    import app as root_app
    # Переносим объекты app и limiter в текущий модуль
    mod.app = root_app.app
    mod.limiter = root_app.limiter
    logger.info("Глобальные app и limiter успешно переданы в модуль маршрутов")
except ImportError:
    logger.error("Не удалось импортировать app.py. Маршруты могут работать некорректно.")

logger.info("Инициализация маршрутов начата")
