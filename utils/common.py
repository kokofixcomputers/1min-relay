# utils/common.py
# Общие утилиты
from .imports import *
from .logger import logger
from .constants import *

def calculate_token(sentence, model="DEFAULT"):
    """
    Рассчитывает количество токенов в строке, используя соответствующую модели токенизацию.
    
    Args:
        sentence (str): Текст для подсчета токенов
        model (str): Модель, для которой необходимо посчитать токены
        
    Returns:
        int: Количество токенов в строке
    """
    if not sentence:
        return 0
        
    try:
        # Выбираем энкодер в зависимости от модели
        encoder_name = "gpt-4"  # Дефолтный энкодер
        
        if model.startswith("mistral"):
            encoder_name = "gpt-4"  # Для Mistral используем OpenAI токенизатор
        elif model in ["gpt-3.5-turbo", "gpt-4"]:
            encoder_name = model
            
        # Получаем токенизатор и считаем токены
        encoding = tiktoken.encoding_for_model(encoder_name)
        tokens = encoding.encode(sentence)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Ошибка при подсчете токенов: {str(e)}. Используем приблизительную оценку.")
        # Приблизительно оцениваем количество токенов как 3/4 количества символов
        return len(sentence) * 3 // 4

def api_request(req_method, url, headers=None, requester_ip=None, data=None,
                files=None, stream=False, timeout=None, json=None, **kwargs):
    """
    Выполняет HTTP-запрос к API с нормализацией URL и обработкой ошибок.
    
    Args:
        req_method (str): Метод запроса (GET, POST, и т.д.)
        url (str): URL для запроса
        headers (dict, optional): Заголовки запроса
        requester_ip (str, optional): IP запрашивающего для логирования
        data (dict/str, optional): Данные для запроса
        files (dict, optional): Файлы для запроса
        stream (bool, optional): Флаг для потоковой передачи данных
        timeout (int, optional): Таймаут запроса в секундах
        json (dict, optional): JSON-данные для запроса
        **kwargs: Дополнительные параметры для requests
        
    Returns:
        Response: Объект ответа от API
    """
    req_url = url.strip()
    logger.debug(f"API request URL: {req_url}")

    # Формируем параметры запроса
    req_params = {k: v for k, v in {
        "headers": headers, 
        "data": data, 
        "files": files, 
        "stream": stream, 
        "json": json
    }.items() if v is not None}
    
    # Добавляем остальные параметры
    req_params.update(kwargs)

    # Определяем, является ли запрос операцией с изображениями
    is_image_operation = False
    if json and isinstance(json, dict):
        operation_type = json.get("type", "")
        if operation_type in [IMAGE_GENERATOR, IMAGE_VARIATOR]:
            is_image_operation = True
            logger.debug(f"Обнаружена операция с изображением: {operation_type}, используем расширенный таймаут")

    # Устанавливаем таймаут в зависимости от типа операции
    req_params["timeout"] = timeout or (MIDJOURNEY_TIMEOUT if is_image_operation else DEFAULT_TIMEOUT)

    # Выполняем запрос
    try:
        response = requests.request(req_method, req_url, **req_params)
        return response
    except Exception as e:
        logger.error(f"Ошибка API запроса: {str(e)}")
        raise

def set_response_headers(response):
    """
    Устанавливает стандартные заголовки для всех ответов API.
    
    Args:
        response: Объект ответа Flask
        
    Returns:
        Response: Модифицированный объект ответа с добавленными заголовками
    """
    response.headers.update({
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "X-Request-ID": str(uuid.uuid4()),
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type, Accept"
    })
    return response

def create_session():
    """
    Создает новую сессию с оптимальными настройками для API запросов.
    
    Returns:
        Session: Настроенная сессия requests
    """
    session = requests.Session()

    # Настраиваем стратегию повторных попыток для всех запросов
    retry_strategy = requests.packages.urllib3.util.retry.Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def safe_temp_file(prefix, request_id=None):
    """
    Безопасно создает временный файл и гарантирует его удаление после использования.

    Args:
        prefix (str): Префикс для имени файла
        request_id (str, optional): ID запроса для логирования

    Returns:
        str: Путь к временному файлу
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp")

    # Создаем временную директорию, если её нет
    os.makedirs(temp_dir, exist_ok=True)

    # Очищаем старые файлы (старше 1 часа)
    try:
        current_time = time.time()
        for old_file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, old_file)
            if os.path.isfile(file_path) and (current_time - os.path.getmtime(file_path) > 3600):
                try:
                    os.remove(file_path)
                    logger.debug(f"[{request_id}] Удален старый временный файл: {file_path}")
                except Exception as e:
                    logger.warning(f"[{request_id}] Не удалось удалить старый временный файл {file_path}: {str(e)}")
    except Exception as e:
        logger.warning(f"[{request_id}] Ошибка при очистке старых временных файлов: {str(e)}")

    # Создаем новый временный файл
    temp_file_path = os.path.join(temp_dir, f"{prefix}_{request_id}_{random_string}")
    return temp_file_path

def ERROR_HANDLER(code, model=None, key=None):
    """
    Обработчик ошибок в формате совместимом с OpenAI API.
    
    Args:
        code (int): Внутренний код ошибки
        model (str, optional): Имя модели (для ошибок, связанных с моделями)
        key (str, optional): API ключ (для ошибок аутентификации)
        
    Returns:
        tuple: (JSON с ошибкой, HTTP-код ответа)
    """
    # Словарь кодов ошибок
    error_codes = {
        1002: {
            "message": f"The model {model} does not exist.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_found",
            "http_code": 400,
        },
        1020: {
            "message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.",
            "type": "authentication_error",
            "param": None,
            "code": "invalid_api_key",
            "http_code": 401,
        },
        1021: {
            "message": "Invalid Authentication",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
            "http_code": 401,
        },
        1212: {
            "message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_supported",
            "http_code": 400,
        },
        1044: {
            "message": f"This model does not support image inputs.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_supported",
            "http_code": 400,
        },
        1412: {
            "message": f"No message provided.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "invalid_request_error",
            "http_code": 400,
        },
        1423: {
            "message": f"No content in last message.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "invalid_request_error",
            "http_code": 400,
        },
    }
    
    # Получаем данные об ошибке или используем данные по умолчанию
    error_data = error_codes.get(code, {
        "message": f"Unknown error (code: {code})",
        "type": "unknown_error",
        "param": None,
        "code": None,
        "http_code": 400
    })
    
    # Удаляем http_code из данных ответа
    http_code = error_data.pop("http_code", 400)
    
    logger.error(f"Ошибка при обработке запроса пользователя. Код ошибки: {code}")
    return jsonify({"error": error_data}), http_code

def handle_options_request():
    """
    Обработчик OPTIONS запросов для CORS.
    
    Returns:
        tuple: (Объект ответа, HTTP-код ответа 204)
    """
    response = make_response()
    response.headers.update({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Allow-Methods": "POST, OPTIONS"
    })
    return response, 204

def split_text_for_streaming(text, chunk_size=6):
    """
    Разбивает текст на небольшие части для эмуляции потокового вывода.

    Args:
        text (str): Текст для разбивки
        chunk_size (int): Примерный размер частей в словах

    Returns:
        list: Список частей текста
    """
    if not text:
        return [""]

    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return [text]

    # Группируем предложения в чанки
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words_in_sentence = len(sentence.split())

        # Если текущий чанк пустой или добавление предложения не превысит лимит слов
        if not current_chunk or current_word_count + words_in_sentence <= chunk_size:
            current_chunk.append(sentence)
            current_word_count += words_in_sentence
        else:
            # Формируем чанк и начинаем новый
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words_in_sentence

    # Добавляем последний чанк, если он не пустой
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks or [text]
