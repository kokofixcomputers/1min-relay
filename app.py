# version 1.0.5 #increment every time you make changes
# 2025-04-07 10:00 #change to actual date and time every time you make changes

# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *

# Инициализируем Flask-приложение
app = Flask(__name__)

# Параметры порта и другие настройки окружения
PORT = int(os.getenv("PORT", 5001))

# Глобальные переменные
MEMORY_STORAGE = {}
MEMCACHED_CLIENT = None

# Инициализируем memcached
try:
    from utils.memcached import check_memcached_connection, delete_all_files_task, safe_memcached_operation
    memcached_available, memcached_uri = check_memcached_connection()
except ImportError as ie:
    logger.error(f"Модуль memcached не найден: {str(ie)}")
    memcached_available = False
    memcached_uri = None
    # Создаем заглушки функций
    def delete_all_files_task():
        logger.warning("Задача удаления файлов отключена (memcached недоступен)")
        
    def safe_memcached_operation(operation, key, value=None, expiry=3600):
        logger.warning(f"Memcached операция {operation} недоступна: модуль не импортирован")
        return None
        
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
            # Сначала пробуем Pymemcache
            from pymemcache.client.base import Client
            
            # Извлекаем хост и порт из URI
            if memcached_uri.startswith('memcached://'):
                host_port = memcached_uri.replace('memcached://', '')
            else:
                host_port = memcached_uri
            
            # Разделяем хост и порт для Pymemcache
            if ':' in host_port:
                host, port = host_port.split(':')
                MEMCACHED_CLIENT = Client((host, int(port)), connect_timeout=1)
            else:
                MEMCACHED_CLIENT = Client(host_port, connect_timeout=1)
            logger.info(f"Клиент Memcached инициализирован через pymemcache: {memcached_uri}")
        except (ImportError, AttributeError, Exception) as e:
            logger.error(f"Ошибка инициализации pymemcache клиента: {str(e)}")
            try:
                # Если не получилось, пробуем Python-Memcache
                if memcached_uri.startswith('memcached://'):
                    host_port = memcached_uri.replace('memcached://', '')
                else:
                    host_port = memcached_uri
                
                MEMCACHED_CLIENT = memcache.Client([host_port], debug=0)
                logger.info(f"Клиент Memcached инициализирован через python-memcached: {memcached_uri}")
            except (ImportError, AttributeError, Exception) as e:
                logger.error(f"Ошибка инициализации memcache клиента: {str(e)}")
                logger.warning(f"Не удалось инициализировать клиент memcached. Хранение сессий отключено.")
    else:
        # Используется для ограничения запросов без memcached
        limiter = Limiter(
            get_remote_address,
            app=app,
        )
        logger.info("Memcached недоступен, хранение сессий отключено")
else:
    # Создаем заглушку для limiter
    class MockLimiter:
        def limit(self, limit_value):
            def decorator(f):
                return f
            return decorator
    
    limiter = MockLimiter()
    logger.info("Flask-limiter не установлен. Используется заглушка для limiter.")

# Читаем переменные окружения
one_min_models_env = os.getenv(
    "SUBSET_OF_ONE_MIN_PERMITTED_MODELS"
)  # напр. "mistral-nemo,gpt-4o,deepseek-chat"
permit_not_in_available_env = os.getenv(
    "PERMIT_MODELS_FROM_SUBSET_ONLY"
)  # напр. "True" или "False"

# Разбираем переменные окружения или используем значения по умолчанию
if one_min_models_env:
    SUBSET_OF_ONE_MIN_PERMITTED_MODELS = one_min_models_env.split(",")

if permit_not_in_available_env and permit_not_in_available_env.lower() == "true":
    PERMIT_MODELS_FROM_SUBSET_ONLY = True

# Объединяем в единый список
AVAILABLE_MODELS = []
AVAILABLE_MODELS.extend(SUBSET_OF_ONE_MIN_PERMITTED_MODELS)

# Добавляем кэш для отслеживания обработанных изображений
# Для каждого запроса храним уникальный идентификатор изображения и его путь
IMAGE_CACHE = {}

# Импортируем вспомогательные функции для routes
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request, safe_temp_file, calculate_token

# Импортируем все маршруты сразу
from routes import *

# Добавляем лог о завершении инициализации в одном месте
logger.info("Инициализация глобальных переменных завершена, app и limiter будут доступны в routes")
logger.info("Все модули маршрутов успешно импортированы")

# Основной код запуска сервера
if __name__ == "__main__":
    # Запуск задачи удаления файлов
    try:
        delete_all_files_task()
    except Exception as e:
        logger.error(f"Ошибка при запуске задачи удаления файлов: {str(e)}")
    
    # Запуск приложения
    internal_ip = socket.gethostbyname(socket.gethostname())
    try:
        response = requests.get("https://api.ipify.org")
        public_ip = response.text
    except Exception as e:
        logger.error(f"Не удалось получить публичный IP: {str(e)}")
        public_ip = "не найден"
    
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
    
    serve(
        app, host="0.0.0.0", port=PORT, threads=6
    )  # Threads по умолчанию 4. Мы используем 6 для повышения производительности и обработки нескольких запросов одновременно.
