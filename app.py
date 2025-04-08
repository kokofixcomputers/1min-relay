# version 1.0.9 #increment every time you make changes
# 2025-04-08 12:00 #change to actual date and time every time you make changes

# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *

# Инициализируем Flask-приложение
app = Flask(__name__)

# Параметры порта и другие настройки окружения
PORT = int(os.getenv("PORT", DEFAULT_PORT))

# Глобальные переменные
MEMORY_STORAGE = {}
MEMCACHED_CLIENT = None
IMAGE_CACHE = {}

# Инициализируем memcached
try:
    from utils.memcached import check_memcached_connection, delete_all_files_task, safe_memcached_operation, set_global_refs
    memcached_available, memcached_uri = check_memcached_connection()
except ImportError as ie:
    logger.error(f"Модуль memcached не найден: {str(ie)}")
    memcached_available = False
    memcached_uri = None
    # Создаем заглушки функций
    def delete_all_files_task():
        logger.warning("Задача удаления файлов отключена (memcached недоступен)")
    def safe_memcached_operation(operation, key, value=None, expiry=MEMCACHED_DEFAULT_EXPIRY):
        logger.warning(f"Memcached операция {operation} недоступна: модуль не импортирован")
        return None
    def set_global_refs(memcached_client=None, memory_storage=None):
        logger.warning("Функция set_global_refs недоступна: модуль не импортирован")
        
# Инициализация лимитера запросов
if LIMITER_AVAILABLE:
    if memcached_available:
        limiter = Limiter(
            get_remote_address,
            app=app,
            storage_uri=memcached_uri,
        )
        # Инициализация клиента Memcache
        try:
            # Извлекаем хост и порт из URI
            host_port = memcached_uri.replace(MEMCACHED_URI_PREFIX, '') if memcached_uri.startswith(MEMCACHED_URI_PREFIX) else memcached_uri
            
            # Разделяем хост и порт
            if ':' in host_port:
                host, port = host_port.split(':')
                port = int(port)
            else:
                host, port = host_port, MEMCACHED_DEFAULT_PORT
                
            # Пробуем сначала Pymemcache, затем Python-Memcache
            try:
                from pymemcache.client.base import Client
                MEMCACHED_CLIENT = Client((host, port), connect_timeout=MEMCACHED_CONNECT_TIMEOUT)
                logger.info(f"Клиент Memcached инициализирован через pymemcache: {memcached_uri}")
            except Exception:
                MEMCACHED_CLIENT = memcache.Client([f"{host}:{port}"], debug=0)
                logger.info(f"Клиент Memcached инициализирован через python-memcached: {memcached_uri}")
        except Exception as e:
            logger.error(f"Ошибка инициализации клиента memcached: {str(e)}")
            logger.warning("Не удалось инициализировать клиент memcached. Хранение сессий отключено.")
    else:
        # Используется для ограничения запросов без memcached
        limiter = Limiter(get_remote_address, app=app)
        logger.info("Memcached недоступен, хранение сессий отключено")
else:
    limiter = MockLimiter()
    logger.info("Flask-limiter не установлен. Используется заглушка для limiter.")

# Устанавливаем глобальные ссылки в модуле memcached
set_global_refs(MEMCACHED_CLIENT, MEMORY_STORAGE)

# Читаем переменные окружения для моделей
one_min_models_env = os.getenv("SUBSET_OF_ONE_MIN_PERMITTED_MODELS")
permit_not_in_available_env = os.getenv("PERMIT_MODELS_FROM_SUBSET_ONLY")

# Разбираем переменные окружения или используем значения по умолчанию
if one_min_models_env:
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS = one_min_models_env.split(",")

if permit_not_in_available_env and permit_not_in_available_env.lower() == "true":
    PERMIT_MODELS_FROM_SUBSET_ONLY = True

# Объединяем в единый список доступных моделей
AVAILABLE_MODELS = []
AVAILABLE_MODELS.extend(SUBSET_OF_ONE_MIN_PERMITTED_MODELS)

# Импортируем вспомогательные функции для routes
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request, safe_temp_file, calculate_token

# Импортируем все маршруты сразу
from routes import *

# Добавляем лог о завершении инициализации
logger.info("Инициализация глобальных переменных завершена, app и limiter будут доступны в routes")
logger.info("Все модули маршрутов успешно импортированы")

# Основной код запуска сервера
if __name__ == "__main__":
    # Запуск задачи удаления файлов
    try:
        delete_all_files_task()
    except Exception as e:
        logger.error(f"Ошибка при запуске задачи удаления файлов: {str(e)}")
    
    # Получение IP-адресов
    internal_ip = socket.gethostbyname(socket.gethostname())
    try:
        public_ip = requests.get("https://api.ipify.org").text
    except Exception as e:
        logger.error(f"Не удалось получить публичный IP: {str(e)}")
        public_ip = "не найден"
    
    # Вывод информации о запуске сервера
    logger.info(
        f"""{printedcolors.Color.fg.lightcyan}  
Сервер готов к работе:
Внутренний IP: {internal_ip}:{PORT}
Публичный IP: {public_ip} (только если вы настроили проброс портов на роутере)
Введите этот URL в клиенты OpenAI, поддерживающие пользовательские эндпоинты:
{internal_ip}:{PORT}/v1
Если не работает, попробуйте:
{internal_ip}:{PORT}/v1/chat/completions
{printedcolors.Color.reset}"""
    )
    
    # Запуск сервера
    serve(app, host=DEFAULT_HOST, port=PORT, threads=DEFAULT_THREADS)
