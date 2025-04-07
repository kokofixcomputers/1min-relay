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

# Экспортируем общие функции
from .functions.shared_func import *

# Экспортируем функции для текстовых моделей
from .functions.txt_func import *

# Экспортируем функции для изображений
from .functions.img_func import *

# Экспортируем функции для аудио
from .functions.audio_func import *

# Экспортируем функции для файлов
from .functions.file_func import *





