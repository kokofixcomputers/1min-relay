# Инициализируем logger для всех модулей routes
import logging
logger = logging.getLogger("1min-relay")

# Импортируем всё содержимое модулей
from .text import *
from .images import *
from .audio import *
from .files import *
