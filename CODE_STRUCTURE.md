# code structure

разбивка на логические модули:
---
1. [app.py](https://github.com/chelaxian/1min-relay/blob/main/app.py) (основной файл)
```python
# Инициализация
app = Flask(__name__)
# ...

# Основные настройки
if __name__ == "__blob/main__":
    # Логирование старта сервера
    # ...
    # Запуск сервера
    serve(app, host="0.0.0.0", port=PORT, threads=6)
```
---
2. [utils/common.py](https://github.com/chelaxian/1min-relay/blob/main/utils/common.py)
```python
# Общие утилиты
def calculate_token(sentence, model="DEFAULT"):
    # ...

def api_request(req_method, url, headers=None, requester_ip=None, data=None, files=None, stream=False, timeout=None, json=None, **kwargs):
    # ...

def set_response_headers(response):
    # ...

def create_session():
    # ...

def safe_temp_file(prefix, request_id=None):
    # ...

def ERROR_HANDLER(code, model=None, key=None):
    # ...

def handle_options_request():
    # ...

def split_text_for_streaming(text, chunk_size=6):
    # ...
```
---
3. [utils/constants.py](https://github.com/chelaxian/1min-relay/blob/main/utils/constants.py)
```python
ONE_MIN_API_URL = "https://api.1min.ai/api/features"
PORT = 5000
# другие глобальные переменные...
```
---
4. [utils/imports.py](https://github.com/chelaxian/1min-relay/blob/main/utils/imports.py)
```python
# Стандартные библиотеки Python
import base64
# Библиотеки Flask и зависимости
from flask import Flask, request, jsonify, make_response, Response, redirect, url_for
# другие импорты...
```
---
5. [utils/logger.py](https://github.com/chelaxian/1min-relay/blob/main/utils/logger.py)
```python
# Создаем логгер
logger = logging.getLogger("1min-relay")
logger.setLevel(logging.DEBUG)
# ...
```
---
6. [utils/memcached.py](https://github.com/chelaxian/1min-relay/blob/main/utils/memcached.py)
```python
# Функции для работы с Memcached
def check_memcached_connection():
    # ...

def set_global_refs(memcached_client=None, memory_storage=None):
    # ...

def safe_memcached_operation(operation, key, value=None, expiry=3600):
    # ...

def delete_all_files_task():
    # ...
```
---
7. [routes/functions.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions.py)
```python
# Общие утилиты для маршрутов - реэкспортирует функции из субмодулей

# Импортируем все функции из подмодулей
from .functions.shared_func import *
from .functions.txt_func import *
from .functions.img_func import *
from .functions.audio_func import *
from .functions.file_func import *
```
---
8. [routes/functions/shared_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/shared_func.py)
```python
# Общие функции аутентификации и форматирования ответов

def validate_auth(request, request_id=None):
    # ...

def handle_api_error(response, api_key=None, request_id=None):
    # ...

def format_openai_response(content, model, request_id=None, prompt_tokens=0):
    # ...

def format_image_response(image_urls, request_id=None, model=None):
    # ...

def stream_response(response, request_data, model, prompt_tokens, session=None):
    # ...
```
---
9. [routes/functions/txt_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/txt_func.py)
```python
# Функции для работы с текстовыми моделями

def prepare_chat_payload(model, messages, request_data, request_id=None):
    # ...

def format_conversation_history(messages, new_input):
    # ...

def get_model_capabilities(model):
    # ...

def prepare_payload(request_data, model, all_messages, image_paths=None, request_id=None):
    # ...

def transform_response(one_min_response, request_data, prompt_token):
    # ...

def emulate_stream_response(full_content, request_data, model, prompt_tokens):
    # ...
```
---
10. [routes/functions/img_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/img_func.py)
```python
# Функции для работы с изображениями

def get_full_url(url, asset_host="https://asset.1min.ai"):
    # ...

def build_generation_payload(model, prompt, request_data, negative_prompt, aspect_ratio, size, mode, request_id):
    # ...

def extract_image_urls_from_response(response_json, request_id):
    # ...

def extract_image_urls(response_data, request_id=None):
    # ...

def prepare_image_payload(model, prompt, request_data, image_paths=None, request_id=None):
    # ...

def parse_aspect_ratio(prompt, model, request_data, request_id=None):
    # ...

def retry_image_upload(image_url, api_key, request_id=None):
    # ...

def create_image_variations(image_url, user_model, n, aspect_width=None, aspect_height=None, mode=None, request_id=None):
    # ...
```
---
11. [routes/functions/audio_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/audio_func.py)
```python
# Функции для работы с аудио

def upload_audio_file(audio_file, api_key, request_id):
    # ...

def try_models_in_sequence(models_to_try, payload_func, api_key, request_id):
    # ...

def extract_text_from_response(response_data, request_id):
    # ...

def prepare_models_list(requested_model, available_models):
    # ...

def get_audio_from_url(audio_url, request_id):
    # ...

def extract_audio_url(response_data, request_id):
    # ...
```
---
12. [routes/functions/file_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/file_func.py)
```python
# Функции для работы с файлами

def get_user_files(api_key, request_id=None):
    # ...

def save_user_files(api_key, files, request_id=None):
    # ...

def create_temp_file(file_data, suffix=".tmp", request_id=None):
    # ...

def upload_asset(file_data, filename, mime_type, api_key, request_id=None, file_type=None):
    # ...

def get_mime_type(filename):
    # ...

def format_file_response(file_info, file_id=None, purpose="assistants", status="processed"):
    # ...

def create_api_response(data, request_id=None):
    # ...

def find_conversation_id(response_data, request_id=None):
    # ...

def find_file_by_id(user_files, file_id):
    # ...

def create_conversation_with_files(file_ids, title, model, api_key, request_id=None):
    # ...
```
---
13. [routes/text.py](https://github.com/chelaxian/1min-relay/blob/main/routes/text.py)
```python
# Маршруты для текстовых моделей
@app.route("/", methods=["GET", "POST"])
def index():
    # ...

@app.route("/v1/models")
@limiter.limit("60 per minute")
def models():
    # ...

@app.route("/v1/chat/completions", methods=["POST"])
@limiter.limit("60 per minute")
def conversation():
    # ...

@app.route("/v1/assistants", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def create_assistant():
    # ...
```
---
14. [routes/images.py](https://github.com/chelaxian/1min-relay/blob/main/routes/images.py)
```python
# Маршруты для работы с изображениями
@app.route("/v1/images/generations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def generate_image():
    # ...

@app.route("/v1/images/variations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
@cross_origin()
def image_variations():
    # ...
```
---
15. [routes/audio.py](https://github.com/chelaxian/1min-relay/blob/main/routes/audio.py)
```python
# Маршруты для работы с аудио
@app.route("/v1/audio/transcriptions", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_transcriptions():
    # ...

@app.route("/v1/audio/translations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_translations():
    # ...

@app.route("/v1/audio/speech", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def text_to_speech():
    # ...
```
---
16. [routes/files.py](https://github.com/chelaxian/1min-relay/blob/main/routes/files.py)
```python
# Маршруты для работы с файлами
@app.route("/v1/files", methods=["GET", "POST", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_files():
    # ...

@app.route("/v1/files/<file_id>", methods=["GET", "DELETE", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file(file_id):
    # ...

@app.route("/v1/files/<file_id>/content", methods=["GET", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file_content(file_id):
    # ...
```
---

