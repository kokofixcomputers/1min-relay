# Функции для работы с Memcached
import hashlib
import json
import logging
import os
import random
import socket
import string
import time
import traceback
from threading import Thread
import uuid

import requests

# Создаем логгер
logger = logging.getLogger("1min-relay")

# Импорт констант
from utils.constants import ONE_MIN_ASSET_URL

# Импортируем необходимые функции
from utils.common import api_request

# Определение глобальной переменной, будет инициализирована позже
MEMCACHED_CLIENT = None
MEMORY_STORAGE = {}

def check_memcached_connection():
    """Проверяет доступность Memcached.
    
    Проверяет настройки окружения для подключения к Memcached и пытается 
    установить соединение для проверки доступности.
    
    Returns:
        tuple: (доступность (bool), URI (str))
    """
    # Читаем конфигурацию мемкэша из переменных окружения
    memcached_host = os.getenv("MEMCACHED_HOST", "memcached")  # По умолчанию имя сервиса в docker-compose
    memcached_port = os.getenv("MEMCACHED_PORT", "11211")
    
    # Формируем URI
    memcached_uri = f"memcached://{memcached_host}:{memcached_port}"
    logger.info(f"Memcached configuration: {memcached_uri}")
    
    # Пытаемся подключиться
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)  # Увеличиваем таймаут для надежности
        try:
            logger.debug(f"Attempting to connect to Memcached at {memcached_host}:{memcached_port}")
            s.connect((memcached_host, int(memcached_port)))
            s.close()
            # Успешно подключились
            logger.info(f"Successfully connected to Memcached at {memcached_host}:{memcached_port}")
            return True, memcached_uri
        except (socket.error, socket.timeout) as e:
            # Если не удалось подключиться по имени сервиса, пробуем localhost
            if memcached_host != "localhost":
                logger.warning(f"Failed to connect to Memcached at {memcached_host}:{memcached_port}, trying localhost")
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect(("localhost", int(memcached_port)))
                    s.close()
                    logger.info(f"Successfully connected to Memcached at localhost:{memcached_port}")
                    return True, f"memcached://localhost:{memcached_port}"
                except (socket.error, socket.timeout) as e2:
                    logger.warning(f"Failed to connect to Memcached at localhost: {str(e2)}")
                    return False, memcached_uri
            else:
                logger.warning(f"Failed to connect to Memcached: {str(e)}")
                return False, memcached_uri
    except Exception as e:
        logger.error(f"Unexpected error while connecting to Memcached: {str(e)}")
        return False, memcached_uri

logger.info(
    """
  _ __  __ _      ___     _           
 / |  \/  (_)_ _ | _ \___| |__ _ _  _ 
 | | |\/| | | ' \|   / -_) / _` | || |
 |_|_|  |_|_|_||_|_|_\___|_\__,_|\_, |
                                 |__/ """
)

def safe_memcached_operation(operation, key, value=None, expiry=None):
    """Безопасно выполняет операции с Memcached.
    
    Выполняет операции с Memcached с обработкой ошибок и поддержкой резервного
    варианта использования локального хранилища.
    
    Args:
        operation (str): Тип операции ('get', 'set', 'delete', 'list')
        key (str): Ключ для операции
        value (any, optional): Значение для сохранения
        expiry (int, optional): Время истечения в секундах
        
    Returns:
        any: Результат операции (зависит от типа операции)
    """
    global MEMCACHED_CLIENT, MEMORY_STORAGE
    
    # Резервный вариант при отсутствии Memcached
    if MEMCACHED_CLIENT is None:
        if operation == "get":
            return MEMORY_STORAGE.get(key)
        elif operation == "set":
            MEMORY_STORAGE[key] = value
            return True
        elif operation == "delete":
            if key in MEMORY_STORAGE:
                del MEMORY_STORAGE[key]
            return True
        elif operation == "list":
            return list(MEMORY_STORAGE.keys())
        else:
            return False
    
    # Используем Memcached
    try:
        if operation == "get":
            value = MEMCACHED_CLIENT.get(key)
            
            # Пытаемся декодировать JSON если это строка
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8')
                except UnicodeDecodeError:
                    pass
                
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
                
            return value
            
        elif operation == "set":
            # Преобразуем dict или list в JSON
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
                
            if expiry:
                return MEMCACHED_CLIENT.set(key, value, time=expiry)
            else:
                return MEMCACHED_CLIENT.set(key, value)
                
        elif operation == "delete":
            return MEMCACHED_CLIENT.delete(key)
            
        elif operation == "list":
            # Достаем все ключи, которые соответствуют паттерну
            # Этот метод специфичен для разных библиотек, поэтому 
            # возвращаем пустой список при отсутствии функционала
            try:
                return MEMCACHED_CLIENT.get_multi(["*"]).keys()
            except (AttributeError, NotImplementedError):
                return []
                
        else:
            return False
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in Memcached operation {operation}: {str(e)}")
        logger.debug(error_traceback)
        
        # Если произошла ошибка с Memcached, возвращаемся к локальному хранилищу
        if operation == "get":
            return MEMORY_STORAGE.get(key)
        elif operation == "set":
            MEMORY_STORAGE[key] = value
            return True
        elif operation == "delete":
            if key in MEMORY_STORAGE:
                del MEMORY_STORAGE[key]
            return True
        elif operation == "list":
            return list(MEMORY_STORAGE.keys())
        else:
            return False

def delete_all_files_task():
    """Периодически удаляет файлы пользователей.
    
    Запускает задачу, которая каждые 30 минут удаляет файлы, 
    загруженные пользователями. Работает в отдельном потоке.
    """
    def task():
        while True:
            try:
                logger.info("Starting periodic file cleanup")
                
                # Получаем список всех ключей пользователей
                user_keys = safe_memcached_operation("list", "*")
                
                # Для каждого пользовательского ключа
                for key in user_keys:
                    if key.startswith("user_"):
                        try:
                            # Получаем данные пользователя
                            user_data = safe_memcached_operation("get", key)
                            
                            if not user_data or not isinstance(user_data, dict):
                                continue
                                
                            # Проверяем, есть ли у пользователя файлы
                            if "files" in user_data and isinstance(user_data["files"], list):
                                # Перебираем файлы пользователя
                                for file_info in user_data["files"]:
                                    if not isinstance(file_info, dict):
                                        continue
                                        
                                    # Проверяем, есть ли информация о файле
                                    if "file_id" in file_info and "createdAt" in file_info:
                                        file_id = file_info["file_id"]
                                        created_at = file_info["createdAt"]
                                        
                                        # Проверяем, прошло ли более 24 часов с момента создания файла
                                        if time.time() - created_at > 24 * 60 * 60:
                                            # Удаляем файл через API
                                            try:
                                                # Импортируем функцию только когда она нужна
                                                from utils.common import api_request
                                                
                                                delete_url = f"{ONE_MIN_ASSET_URL}/files/{file_id}"
                                                response = api_request(delete_url, "DELETE")
                                                
                                                if hasattr(response, "status_code") and response.status_code == 200:
                                                    logger.info(f"Successfully deleted file {file_id}")
                                                    
                                                    # Удаляем запись о файле из данных пользователя
                                                    user_data["files"].remove(file_info)
                                                    safe_memcached_operation("set", key, user_data)
                                                else:
                                                    logger.warning(f"Failed to delete file {file_id}")
                                            except Exception as e:
                                                logger.error(f"Error deleting file {file_id}: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error processing user key {key}: {str(e)}")
                
                logger.info("Completed periodic file cleanup")
            except Exception as e:
                logger.error(f"Error in delete_all_files_task: {str(e)}")
                
            # Спим 30 минут перед следующей итерацией
            time.sleep(30 * 60)
    
    # Запускаем в отдельном потоке
    thread = Thread(target=task, daemon=True)
    thread.start()
    return thread

