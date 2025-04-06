# code structure

разбивка на логические модули:
---
1. [app.py](https://github.com/chelaxian/1min-relay/blob/test/app.py) (основной файл)
```python
# Инициализация
app = Flask(__name__)
CORS(app)
limiter = Limiter(...)

# Основные настройки
if __name__ == "__main__":
    # Launch the task of deleting files
    delete_all_files_task()

    # Логирование старта сервера
    # ...

    # Запуск сервера
    serve(app, host="0.0.0.0", port=PORT, threads=6)
```
---
2. [utils/common.py](https://github.com/chelaxian/1min-relay/blob/test/utils/common.py)
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
3. [utils/constants.py](https://github.com/chelaxian/1min-relay/blob/test/utils/constants.py)
```python
PORT = 5000
# другие глобальные переменные...
```
---
4. [utils/imports.py](https://github.com/chelaxian/1min-relay/blob/test/utils/imports.py)
```python
from flask import Flask, request, jsonify, make_response, redirect
from flask_cors import CORS, cross_origin
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from waitress import serve
# другие импорты...
...
```
---
5. [utils/logger.py](https://github.com/chelaxian/1min-relay/blob/test/utils/logger.py)
```python
# Создаем логгер
logger = logging.getLogger("1min-relay")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Предотвращаем дублирование логов
# ...
```
---
6. [utils/memcached.py](https://github.com/chelaxian/1min-relay/blob/test/utils/memcached.py)
```python
# Функции для работы с Memcached
def check_memcached_connection():
    # ...

def safe_memcached_operation(operation, key, value=None, expiry=3600):
    # ...

def delete_all_files_task():
    # ...
```
---
7. [routes/text.py](https://github.com/chelaxian/1min-relay/blob/test/routes/text.py)
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

# Вспомогательные функции для текстовых моделей
def format_conversation_history(messages, new_input):
    # ...

def get_model_capabilities(model):
    # ...

def prepare_payload(request_data, model, all_messages, image_paths=None, request_id=None):
    # ...

def transform_response(one_min_response, request_data, prompt_token):
    # ...

def stream_response(response, request_data, model, prompt_tokens, session=None):
    # ...

def emulate_stream_response(full_content, request_data, model, prompt_tokens):
    # ...
```
---
8. [routes/images.py](https://github.com/chelaxian/1min-relay/blob/test/routes/images.py)
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

# Вспомогательные функции для изображений
def parse_aspect_ratio(prompt, model, request_data, request_id=None):
    # ...

def retry_image_upload(image_url, api_key, request_id=None):
    # ...

def create_image_variations(image_url, user_model, n, aspect_width=None, aspect_height=None, mode=None, request_id=None):
    # ...
```
---
9. [routes/audio.py](https://github.com/chelaxian/1min-relay/blob/test/routes/audio.py)
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
10. [routes/files.py](https://github.com/chelaxian/1min-relay/blob/test/routes/files.py)
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

@app.route("/v1/files", methods=["POST"])
@limiter.limit("60 per minute")
def upload_file():
    # ...

# Вспомогательные функции для работы с файлами
def upload_document(file_data, file_name, api_key, request_id=None):
    # ...

def create_conversation_with_files(file_ids, title, model, api_key, request_id=None):
    # ...
```
---

