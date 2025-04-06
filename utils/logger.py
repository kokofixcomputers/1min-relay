# utils/logger.py
# Модуль для логгирования
import logging
import sys
import os
from datetime import datetime

# Создаем директорию для логов, если её нет
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
if not os.path.exists(log_dir):
    try:
        os.makedirs(log_dir)
    except Exception:
        # Если не удалось создать директорию, продолжаем без файлового логирования
        log_dir = None

# Создаем логгер
logger = logging.getLogger("1min-relay")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Предотвращаем дублирование логов

# Форматтер для вывода цветного текста в консоль
class ColoredFormatter(logging.Formatter):
    """Форматтер, который добавляет цвета в логи в консоли"""
    
    # Цвета ANSI
    COLORS = {
        'DEBUG': '\033[36m',     # Голубой
        'INFO': '\033[32m',      # Зеленый
        'WARNING': '\033[33m',   # Желтый
        'ERROR': '\033[31m',     # Красный
        'CRITICAL': '\033[35m',  # Пурпурный
        'RESET': '\033[0m'       # Сброс цвета
    }
    
    def format(self, record):
        # Проверяем, поддерживает ли терминал цвета
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                record.msg = f"{self.COLORS[levelname]}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)

# Создаем консольный обработчик с цветным форматтером
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
color_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(color_formatter)
logger.addHandler(console_handler)

# Если директория для логов существует, добавляем файловый обработчик
if log_dir:
    try:
        log_file = os.path.join(log_dir, f"relay_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Не удалось настроить файловое логирование: {str(e)}")

# Функция для вывода ASCII-баннера
def print_banner():
    logger.info(
        """
  _ __  __ _      ___     _           
 / |  \/  (_)_ _ | _ \___| |__ _ _  _ 
 | | |\/| | | ' \|   / -_) / _` | || |
 |_|_|  |_|_|_||_|_|_\___|_\__,_|\_, |
                                 |__/ """
    )

# Выводим баннер при импорте модуля
print_banner() 
