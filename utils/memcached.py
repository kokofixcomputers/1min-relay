# version 1.0.3 #increment every time you make changes
# utils/memcached.py
# Функции для работы с Memcached
from .imports import *
from .logger import logger
from .constants import *

# Объявляем глобальные переменные для хранения ссылок
MEMCACHED_CLIENT_REF = None
MEMORY_STORAGE_REF = None

def check_memcached_connection():
    """
    Проверяет доступность Memcached, сначала локально, затем в Docker

    Returns:
        Tuple: (Bool, Str) - (Доступен ли Memcached, строка подключения или None)
    """
    # Проверяем наличие библиотек для работы с Memcached
    if not MEMCACHED_AVAILABLE:
        logger.warning(
            "Memcached библиотеки не установлены. Используется локальное хранилище для ограничения запросов. Не рекомендуется."
        )
        return False, None
    
    # Функция для проверки подключения по адресу с таймаутом
    def try_memcached_connection(host, port):
        try:
            from pymemcache.client.base import Client
            client = Client((host, port), connect_timeout=MEMCACHED_CONNECT_TIMEOUT, timeout=MEMCACHED_OPERATION_TIMEOUT)
            client.set("test_key", "test_value")
            if client.get("test_key") == b"test_value":
                client.delete("test_key")  # Очистка
                return True
        except Exception as e:
            logger.debug(f"Memcached на {host}:{port} недоступен: {str(e)}")
            return False
        return False
    
    # Сначала проверяем локальный Memcached, используя константы
    if try_memcached_connection(MEMCACHED_DEFAULT_HOST, MEMCACHED_DEFAULT_PORT):
        logger.info(f"Используется локальный Memcached на {MEMCACHED_DEFAULT_HOST}:{MEMCACHED_DEFAULT_PORT}")
        return True, f"{MEMCACHED_URI_PREFIX}{MEMCACHED_DEFAULT_HOST}:{MEMCACHED_DEFAULT_PORT}"
    
    # Если локальный недоступен, проверяем Docker Memcached, используя константы
    if try_memcached_connection(MEMCACHED_DOCKER_HOST, MEMCACHED_DEFAULT_PORT):
        logger.info(f"Используется Memcached в Docker-контейнере на {MEMCACHED_DOCKER_HOST}:{MEMCACHED_DEFAULT_PORT}")
        return True, f"{MEMCACHED_URI_PREFIX}{MEMCACHED_DOCKER_HOST}:{MEMCACHED_DEFAULT_PORT}"

    # Если ни Docker, ни локальный Memcached недоступны
    logger.warning(
        "Memcached недоступен (ни в Docker, ни локально). Используется локальное хранилище для ограничения запросов. Не рекомендуется."
    )
    return False, None


# Устанавливаем ссылки на глобальные объекты из app.py
def set_global_refs(memcached_client=None, memory_storage=None):
    """
    Устанавливает ссылки на глобальные объекты из app.py
    
    Args:
        memcached_client: Клиент Memcached
        memory_storage: Хранилище в памяти
    """
    global MEMCACHED_CLIENT_REF, MEMORY_STORAGE_REF
    MEMCACHED_CLIENT_REF = memcached_client
    MEMORY_STORAGE_REF = memory_storage


# Функция для безопасного доступа к Memcached
def safe_memcached_operation(operation, key, value=None, expiry=MEMCACHED_DEFAULT_EXPIRY):
    """
    Безопасно выполняет операции с Memcached, обрабатывая любые исключения.
    
    Args:
        operation (str): Операция для выполнения ('get', 'set' или 'delete')
        key (str): Ключ для операции
        value (any, optional): Значение для установки (только для операции 'set')
        expiry (int, optional): Время истечения в секундах (только для операции 'set')
    
    Returns:
        Результат операции или None в случае неудачи
    """
    # Функция для работы с локальным хранилищем
    def use_memory_storage():
        if operation == 'get':
            return MEMORY_STORAGE_REF.get(key, None) if MEMORY_STORAGE_REF else None
        elif operation == 'set':
            if MEMORY_STORAGE_REF is not None:
                MEMORY_STORAGE_REF[key] = value
                logger.debug(f"Сохранено в MEMORY_STORAGE: key={key}")
            return True
        elif operation == 'delete':
            if MEMORY_STORAGE_REF is not None and key in MEMORY_STORAGE_REF:
                del MEMORY_STORAGE_REF[key]
                return True
            return False
        return None
    
    if MEMCACHED_CLIENT_REF is None:
        return use_memory_storage()
    
    try:
        if operation == 'get':
            result = MEMCACHED_CLIENT_REF.get(key)
            if isinstance(result, bytes):
                try:
                    return json.loads(result.decode('utf-8'))
                except:
                    return result.decode('utf-8')
            return result
        elif operation == 'set':
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            # Пробуем разные варианты параметров для времени истечения
            exp_params = ['exp', 'exptime', 'expire', 'time']
            for exp_param in exp_params:
                try:
                    return MEMCACHED_CLIENT_REF.set(key, value, **{exp_param: expiry})
                except TypeError as te:
                    if f"unexpected keyword argument '{exp_param}'" in str(te):
                        continue
                    raise
                except Exception:
                    raise
            
            # Если все варианты не подошли, пробуем без параметра времени истечения
            logger.warning(f"Не удалось найти подходящий параметр для времени истечения, используем без параметра")
            return MEMCACHED_CLIENT_REF.set(key, value)
            
        elif operation == 'delete':
            return MEMCACHED_CLIENT_REF.delete(key)
    except Exception as e:
        logger.error(f"Ошибка в операции memcached {operation} на ключе {key}: {str(e)}")
        # При ошибке Memcached, также используем локальное хранилище
        return use_memory_storage()
        
def delete_all_files_task():
    """
    Функция для периодического удаления всех файлов пользователей
    """
    # Проверяем, включена ли автоматическая очистка
    if not FILE_CLEANUP_ENABLED:
        logger.info("Автоматическая очистка файлов отключена")
        return
        
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Запуск запланированной задачи очистки файлов")

    try:
        # Проверка доступности Memcached
        if MEMCACHED_CLIENT_REF is None:
            logger.warning(f"[{request_id}] Memcached недоступен, очистка файлов невозможна")
            return

        # Получаем список всех известных пользователей
        known_users = safe_memcached_operation('get', 'known_users_list') or []
        
        # Конвертируем в список, если получено в другом формате
        if isinstance(known_users, str):
            try:
                known_users = json.loads(known_users)
            except:
                known_users = []
        elif isinstance(known_users, bytes):
            try:
                known_users = json.loads(known_users.decode('utf-8'))
            except:
                known_users = []
                
        if not known_users:
            logger.info(f"[{request_id}] Нет известных пользователей для очистки файлов")
            return
            
        logger.info(f"[{request_id}] Найдено {len(known_users)} пользователей для очистки файлов")
        
        # Обрабатываем каждого пользователя
        for user in known_users:
            user_key = f"user:{user}" if not user.startswith("user:") else user
            api_key = user_key.replace("user:", "")
            
            # Получаем файлы пользователя
            user_files_json = safe_memcached_operation('get', user_key)
            if not user_files_json:
                continue
                
            # Преобразуем данные в список файлов
            user_files = []
            try:
                if isinstance(user_files_json, str):
                    user_files = json.loads(user_files_json)
                elif isinstance(user_files_json, bytes):
                    user_files = json.loads(user_files_json.decode('utf-8'))
                else:
                    user_files = user_files_json
            except:
                continue
                
            if not user_files:
                continue
                
            logger.info(f"[{request_id}] Очистка {len(user_files)} файлов для пользователя {api_key[:8]}...")
            
            # Удаляем каждый файл
            for file_info in user_files:
                file_id = file_info.get("id")
                if file_id:
                    try:
                        from .common import api_request  # Импортируем здесь, чтобы избежать циклической зависимости
                        delete_url = f"{ONE_MIN_ASSET_URL}/{file_id}"
                        headers = {"API-KEY": api_key}

                        delete_response = api_request("DELETE", delete_url, headers=headers)

                        if delete_response.status_code == 200:
                            logger.info(f"[{request_id}] Запланированная очистка: удален файл {file_id}")
                        else:
                            logger.error(
                                f"[{request_id}] Запланированная очистка: не удалось удалить файл {file_id}: {delete_response.status_code}")
                    except Exception as e:
                        logger.error(
                            f"[{request_id}] Запланированная очистка: ошибка при удалении файла {file_id}: {str(e)}")
            
            # Очистка списка файлов пользователя
            safe_memcached_operation('set', user_key, json.dumps([]))
            logger.info(f"[{request_id}] Очищен список файлов для пользователя {api_key[:8]}")
            
    except Exception as e:
        logger.error(f"[{request_id}] Ошибка в запланированной задаче очистки: {str(e)}")

    # Планируем следующее выполнение через заданный интервал
    cleanup_timer = threading.Timer(FILE_CLEANUP_INTERVAL, delete_all_files_task)
    cleanup_timer.daemon = True
    cleanup_timer.start()
    logger.info(f"[{request_id}] Запланирована следующая очистка через {FILE_CLEANUP_INTERVAL} секунд")
