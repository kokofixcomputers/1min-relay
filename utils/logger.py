# utils/logger.py
# Модуль для логгирования
import logging
import coloredlogs

# Create a logger object
logger = logging.getLogger("1min-relay")

# Install coloredlogs with desired log level
coloredlogs.install(level="DEBUG", logger=logger) 
