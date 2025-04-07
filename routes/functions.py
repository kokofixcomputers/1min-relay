# routes/functions.py
# Общие утилиты для маршрутов

# Реэкспортируем основные зависимости
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import (
    ERROR_HANDLER, 
    handle_options_request, 
    set_response_headers, 
    create_session, 
    api_request, 
    safe_temp_file, 
    calculate_token
)
from utils.memcached import safe_memcached_operation

# Импортируем все функции из подмодулей
from .functions.shared_func import *
from .functions.txt_func import *
from .functions.img_func import *
from .functions.audio_func import *
from .functions.file_func import *

# Альтернативный способ: импортировать всё из __init__.py подпакета
# from .functions import *





