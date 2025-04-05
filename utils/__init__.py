# Импортируем логгер перед всеми остальными модулями
from .logger import logger

# Импортируем всё содержимое модулей
from .constants import *
from .common import *
from .memcached import *
