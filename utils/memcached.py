# utils/memcached.py
# Функции для работы с Memcached
from .imports import *
from .logger import logger
from .constants import *

# Глобальные переменные для хранения клиента memcached и в памяти
MEMCACHED_CLIENT = None
MEMORY_STORAGE = {}

def check_memcached_connection():
    """
    Проверяет доступность Memcached, сначала в Docker, затем локально

    Returns:
        Tuple: (Bool, Str) - (Доступен ли Memcached, строка подключения или None)
    """
    # Проверяем наличие библиотек для работы с Memcached
    if not MEMCACHED_AVAILABLE:
        logger.warning(
            "Memcached библиотеки не установлены. Используется локальное хранилище для ограничения запросов. Не рекомендуется."
        )
        return False, None
    
    # Проверяем Docker Memcached (приоритет)
    try:
        from pymemcache.client.base import Client
        client = Client(("memcached", 11211))
        client.set("test_key", "test_value")
        if client.get("test_key") == b"test_value":
            client.delete("test_key")  # Очистка
            logger.info("Используется Memcached в Docker-контейнере")
            return True, "memcached://memcached:11211"
    except Exception as e:
        logger.debug(f"Docker Memcached недоступен: {str(e)}")
    
    # Если Docker недоступен, проверяем локальный Memcached
    try:
        from pymemcache.client.base import Client
        client = Client(("127.0.0.1", 11211))
        client.set("test_key", "test_value")
        if client.get("test_key") == b"test_value":
            client.delete("test_key")  # Очистка
            logger.info("Используется локальный Memcached на 127.0.0.1:11211")
            return True, "memcached://127.0.0.1:11211"
    except Exception as e:
        logger.debug(f"Локальный Memcached недоступен: {str(e)}")

    # Если ни Docker, ни локальный Memcached недоступны
    logger.warning(
        "Memcached недоступен (ни в Docker, ни локально). Используется локальное хранилище для ограничения запросов. Не рекомендуется."
    )
    return False, None


# Функция для безопасного доступа к Memcached
def safe_memcached_operation(operation, key, value=None, expiry=3600):
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
    if MEMCACHED_CLIENT is None:
        # Если Memcached недоступен, используем локальное хранилище
        if operation == 'get':
            return MEMORY_STORAGE.get(key, None)
        elif operation == 'set':
            MEMORY_STORAGE[key] = value
            logger.info(f"Сохранено в MEMORY_STORAGE: key={key}")
            return True
        elif operation == 'delete':
            if key in MEMORY_STORAGE:
                del MEMORY_STORAGE[key]
                return True
            return False
        return None
    
    try:
        if operation == 'get':
            result = MEMCACHED_CLIENT.get(key)
            if isinstance(result, bytes):
                try:
                    return json.loads(result.decode('utf-8'))
                except:
                    return result.decode('utf-8')
            return result
        elif operation == 'set':
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return MEMCACHED_CLIENT.set(key, value, time=expiry)
        elif operation == 'delete':
            return MEMCACHED_CLIENT.delete(key)
    except Exception as e:
        logger.error(f"Ошибка в операции memcached {operation} на ключе {key}: {str(e)}")
        # При ошибке Memcached, также используем локальное хранилище
        if operation == 'get':
            return MEMORY_STORAGE.get(key, None)
        elif operation == 'set':
            MEMORY_STORAGE[key] = value
            logger.info(f"Сохранено в MEMORY_STORAGE из-за ошибки memcached: key={key}")
            return True
        elif operation == 'delete':
            if key in MEMORY_STORAGE:
                del MEMORY_STORAGE[key]
                return True
            return False
        return None
        
def delete_all_files_task():
    """
    Функция для периодического удаления всех файлов пользователей
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Запуск запланированной задачи очистки файлов")

    try:
        # Получаем всех пользователей с файлами из MemcacheD
        if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
            # Получаем все ключи, которые начинаются с "user:"
            try:
                keys = []

                # Вместо сканирования слэбов, используем список известных пользователей
                # который должен сохраняться при загрузке файлов
                known_users = safe_memcached_operation('get', 'known_users_list')
                if known_users:
                    try:
                        if isinstance(known_users, str):
                            user_list = json.loads(known_users)
                        elif isinstance(known_users, bytes):
                            user_list = json.loads(known_users.decode('utf-8'))
                        else:
                            user_list = known_users

                        for user in user_list:
                            user_key = f"user:{user}" if not user.startswith("user:") else user
                            if user_key not in keys:
                                keys.append(user_key)
                    except Exception as e:
                        logger.warning(f"[{request_id}] Не удалось разобрать список известных пользователей: {str(e)}")

                logger.info(f"[{request_id}] Найдено {len(keys)} ключей пользователей для очистки")

                # Удаляем файлы для каждого пользователя
                for user_key in keys:
                    try:
                        api_key = user_key.replace("user:", "")
                        user_files_json = safe_memcached_operation('get', user_key)

                        if not user_files_json:
                            continue

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
                        logger.error(f"[{request_id}] Ошибка при обработке пользователя {user_key}: {str(e)}")
            except Exception as e:
                logger.error(f"[{request_id}] Ошибка получения ключей из memcached: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Ошибка в запланированной задаче очистки: {str(e)}")

    # Планируем следующее выполнение через час
    cleanup_timer = threading.Timer(3600, delete_all_files_task)
    cleanup_timer.daemon = True
    cleanup_timer.start()
    logger.info(f"[{request_id}] Запланирована следующая очистка через 1 час")
