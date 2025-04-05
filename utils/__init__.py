# Инициализируем logger для всех модулей routes
import logging
logger = logging.getLogger("1min-relay")

# Импортируем всё содержимое модулей
from .common import *
from .constants import *
from .memcached import *
