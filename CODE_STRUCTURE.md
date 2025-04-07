# Структура кода проекта

## Основной файл приложения: `1min-relay/app.py`
- Инициализирует Flask-приложение и запускает сервер
- Импортирует все необходимые модули
- Настраивает параметры сервера

## Утилиты

### Общие утилиты: `1min-relay/utils/common.py`
- `ERROR_HANDLER`: Создает стандартизированные ответы с ошибками и соответствующими статус-кодами
- `handle_options_request`: Обрабатывает OPTIONS-запросы для CORS
- `set_response_headers`: Устанавливает заголовки ответов для CORS
- `create_session`: Создает сессию для API-запросов
- `api_request`: Выполняет запросы к внешним API с обработкой ошибок
- `safe_temp_file`: Создает временный файл с правильным управлением ресурсами
- `calculate_token`: Рассчитывает количество токенов в тексте с помощью tiktoken

### Константы: `1min-relay/utils/constants.py`
- `ENDPOINTS`: Словарь конечных точек API
- `ROLES_MAPPING`: Сопоставление ролей для разных моделей
- `MODEL_CAPABILITIES`: Словарь возможностей моделей
- Различные другие константы, используемые в приложении

### Импорты: `1min-relay/utils/imports.py`
- Центральное место для всех импортов стандартных библиотек
- Импорты, используемые в нескольких модулях

### Логгер: `1min-relay/utils/logger.py`
- `logger`: Настроенный экземпляр логгера
- Функции для настройки и использования логгера

### Memcached: `1min-relay/utils/memcached.py`
- `MEMORY_STORAGE`: Словарь для временного хранения
- `safe_memcached_operation`: Безопасно выполняет операции с memcached
- `delete_all_files_task`: Периодически удаляет устаревшие файлы пользователей

## Функции

### Инициализация функций: `1min-relay/routes/functions/__init__.py`
- Четкий экспорт всех необходимых функций из подмодулей
- Группировка и документирование функций по категориям
- Обеспечивает удобный импорт функций в маршрутах

### Общие функции: `1min-relay/routes/functions/shared_func.py`
- `validate_auth`: Проверяет заголовок авторизации
- `handle_api_error`: Стандартизированная обработка ошибок для ответов API
- `format_openai_response`: Форматирует ответы в соответствии с API OpenAI
- `format_image_response`: Форматирует ответы с изображениями в соответствии с API OpenAI
- `stream_response`: Передает ответы API в потоковом режиме
- `get_full_url`: Создает полный URL из относительного пути
- `extract_data_from_api_response`: Общая функция для извлечения данных из API-ответов
- `extract_text_from_response`: Извлекает текст из API-ответов
- `extract_image_urls`: Извлекает URL-адреса изображений из API-ответов
- `extract_audio_url`: Извлекает URL-адрес аудио из API-ответов

### Текстовые функции: `1min-relay/routes/functions/txt_func.py`
- `format_conversation_history`: Форматирует историю разговора для моделей
- `get_model_capabilities`: Получает информацию о возможностях модели
- `prepare_payload`: Подготавливает нагрузку для API-запросов
- `transform_response`: Преобразует ответы API
- `emulate_stream_response`: Эмулирует потоковый ответ

### Функции для изображений: `1min-relay/routes/functions/img_func.py`
- `build_generation_payload`: Создает нагрузку для генерации изображений
- `parse_aspect_ratio`: Анализирует соотношение сторон из входных данных
- `create_image_variations`: Создает вариации изображений
- `build_img2img_payload`: Создает нагрузку для запросов img2img
- `process_image_tool_calls`: Обрабатывает вызовы инструментов для изображений

### Аудио функции: `1min-relay/routes/functions/audio_func.py`
- `upload_audio_file`: Загружает аудиофайлы
- `try_models_in_sequence`: Последовательно пробует разные модели
- `prepare_models_list`: Подготавливает список моделей для попытки
- `prepare_whisper_payload`: Подготавливает нагрузку для API Whisper
- `prepare_tts_payload`: Подготавливает нагрузку для преобразования текста в речь

### Файловые функции: `1min-relay/routes/functions/file_func.py`
- `get_user_files`: Получает файлы пользователя из Memcached
- `save_user_files`: Сохраняет файлы пользователя в Memcached
- `create_temp_file`: Создает временные файлы
- `upload_asset`: Загружает активы на сервер
- `get_mime_type`: Получает MIME-тип файла
- `retry_image_upload`: Повторяет загрузку изображения при сбое
- `format_file_response`: Форматирует ответ о файле в формате OpenAI
- `create_api_response`: Создает HTTP-ответ с правильными заголовками
- `find_conversation_id`: Находит ID разговора в ответе API
- `find_file_by_id`: Находит файл по ID в списке файлов пользователя
- `create_conversation_with_files`: Создает новый разговор с файлами

## Маршруты

### Текстовые маршруты: `1min-relay/routes/text.py`
- `/v1/models`: Возвращает список доступных моделей
- `/v1/chat/completions`: Обрабатывает запросы на завершение чата
- Различные другие конечные точки текстовых моделей

### Маршруты изображений: `1min-relay/routes/images.py`
- `/v1/images/generations`: Генерирует изображения из текста
- `/v1/images/variations`: Создает вариации изображений

### Аудио маршруты: `1min-relay/routes/audio.py`
- `/v1/audio/transcriptions`: Транскрибирует аудио в текст
- `/v1/audio/translations`: Переводит аудио на другой язык
- `/v1/audio/speech`: Преобразует текст в речь

### Файловые маршруты: `1min-relay/routes/files.py`
- `/v1/files`: Обрабатывает загрузку и управление файлами
- `/v1/files/<file_id>`: Получает или удаляет конкретный файл
- `/v1/files/<file_id>/content`: Получает содержимое файла 
