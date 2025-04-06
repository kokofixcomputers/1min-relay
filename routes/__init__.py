# routes/__init__.py
# Импорт маршрутов

# Импортируем логгер
from utils.logger import logger
from utils.imports import *
from utils.constants import *
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request, safe_temp_file, calculate_token

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
    
    # Импортируем все модули маршрутов
    from . import files
    from . import text
    from . import images
    from . import audio
    logger.info("Все модули маршрутов импортированы")
    
    # Обеспечиваем прямой доступ к маршрутам из корневого модуля
    root_app.routes = mod
    logger.info("Модуль маршрутов добавлен в корневой модуль app")
    
except ImportError as e:
    logger.error(f"Не удалось импортировать app.py: {str(e)}. Маршруты могут работать некорректно.")

logger.info("Инициализация маршрутов завершена")
