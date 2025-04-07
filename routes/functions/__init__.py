# routes/functions/__init__.py
# Инициализация субпакета функций

# Экспортируем общие функции
from .shared_func import *

# Экспортируем функции для текстовых моделей
from .txt_func import *

# Экспортируем функции для изображений
from .img_func import *

# Экспортируем функции для аудио
from .audio_func import *

# Экспортируем функции для файлов
from .file_func import *
