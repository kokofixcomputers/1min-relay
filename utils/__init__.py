# version 1.0.1 #increment every time you make changes
# utils/__init__.py
# Инициализация пакета utils, импорт модулей в правильном порядке

# Сначала импортируем централизованный модуль импортов
from .imports import *

# Затем импортируем логгер (баннер уже выводится при импорте logger)
from .logger import logger

# Потом импортируем константы
from .constants import *

# Импортируем функцию для установки глобальных ссылок из memcached
from .memcached import set_global_refs

# Наконец, импортируем общие функции
from .common import (
    ERROR_HANDLER, 
    handle_options_request, 
    set_response_headers, 
    create_session, 
    api_request, 
    safe_temp_file, 
    calculate_token,
    split_text_for_streaming
)

# Примечание: остальные функции из модуля memcached не импортируются здесь напрямую,
# чтобы избежать циклической зависимости. При необходимости их
# следует импортировать в конкретном модуле:
# from utils.memcached import safe_memcached_operation, check_memcached_connection
