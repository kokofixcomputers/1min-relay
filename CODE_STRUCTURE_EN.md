# Code Structure

Division into logical modules:
---
1. [app.py](https://github.com/chelaxian/1min-relay/blob/main/app.py) (blob/main file)
```python
# Initialization
app = Flask(__name__)
# ...

# blob/main settings
if __name__ == "__blob/main__":
    # Logging server start
    # ...
    # Server launch
    serve(app, host="0.0.0.0", port=PORT, threads=6)
```
---
2. [utils/common.py](https://github.com/chelaxian/1min-relay/blob/main/utils/common.py)
```python
# Common utilities
def calculate_token(sentence, model="DEFAULT"):
    # Function to calculate token count in text

def api_request(req_method, url, headers=None, requester_ip=None, data=None, files=None, stream=False, timeout=None, json=None, **kwargs):
    # Function to make API requests with error handling

def set_response_headers(response):
    # Function to set CORS headers on responses

def create_session():
    # Function to create a request session

def safe_temp_file(prefix, request_id=None):
    # Function to create temporary files safely

def ERROR_HANDLER(code, model=None, key=None):
    # Function to handle error codes and return formatted messages

def handle_options_request():
    # Function to handle OPTIONS requests for CORS

def split_text_for_streaming(text, chunk_size=6):
    # Function to split text into chunks for streaming
```
---
3. [utils/constants.py](https://github.com/chelaxian/1min-relay/blob/main/utils/constants.py)
```python
ONE_MIN_API_URL = "https://api.1min.ai/api/features"  # Base API URL
PORT = 5000  # Default port
# other global variables...
```
---
4. [utils/imports.py](https://github.com/chelaxian/1min-relay/blob/main/utils/imports.py)
```python
# Standard Python libraries
import base64
# Flask libraries and dependencies
from flask import Flask, request, jsonify, make_response, Response, redirect, url_for
# other imports...
```
---
5. [utils/logger.py](https://github.com/chelaxian/1min-relay/blob/main/utils/logger.py)
```python
# Create logger
logger = logging.getLogger("1min-relay")
logger.setLevel(logging.DEBUG)
# ...
```
---
6. [utils/memcached.py](https://github.com/chelaxian/1min-relay/blob/main/utils/memcached.py)
```python
# Functions for working with Memcached
def check_memcached_connection():
    # Function to check and initialize memcached connection

def set_global_refs(memcached_client=None, memory_storage=None):
    # Function to set global references to memcached client

def safe_memcached_operation(operation, key, value=None, expiry=3600):
    # Function to safely perform memcached operations

def delete_all_files_task():
    # Function to clean up old files
```
---
7. [routes/functions.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions.py)
```python
# Common utilities for routes - re-exports functions from submodules

# Import all functions from submodules
from .functions.shared_func import *
from .functions.txt_func import *
from .functions.img_func import *
from .functions.audio_func import *
from .functions.file_func import *
```
---
8. [routes/functions/shared_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/shared_func.py)
```python
# Common authentication and response formatting functions

def validate_auth(request, request_id=None):
    # Function to validate API key in requests

def handle_api_error(response, api_key=None, request_id=None):
    # Function to process API errors

def format_openai_response(content, model, request_id=None, prompt_tokens=0):
    # Function to format responses in OpenAI format

def format_image_response(image_urls, request_id=None, model=None):
    # Function to format image responses in OpenAI format

def stream_response(response, request_data, model, prompt_tokens, session=None):
    # Function to stream API responses in OpenAI format
```
---
9. [routes/functions/txt_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/txt_func.py)
```python
# Functions for working with text models

def prepare_chat_payload(model, messages, request_data, request_id=None):
    # Function to prepare payload for chat requests

def format_conversation_history(messages, new_input):
    # Function to format conversation history

def get_model_capabilities(model):
    # Function to determine model capabilities

def prepare_payload(request_data, model, all_messages, image_paths=None, request_id=None):
    # Function to prepare API request payload

def transform_response(one_min_response, request_data, prompt_token):
    # Function to transform API response into OpenAI format

def emulate_stream_response(full_content, request_data, model, prompt_tokens):
    # Function to emulate streaming for models that don't support it
```
---
10. [routes/functions/img_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/img_func.py)
```python
# Functions for working with images

def get_full_url(url, asset_host="https://asset.1min.ai"):
    # Function to build complete image URLs

def build_generation_payload(model, prompt, request_data, negative_prompt, aspect_ratio, size, mode, request_id):
    # Function to build payload for image generation

def extract_image_urls_from_response(response_json, request_id):
    # Function to extract image URLs from API response

def extract_image_urls(response_data, request_id=None):
    # Function to extract image URLs from various response formats

def prepare_image_payload(model, prompt, request_data, image_paths=None, request_id=None):
    # Function to prepare payload for image generation

def parse_aspect_ratio(prompt, model, request_data, request_id=None):
    # Function to parse aspect ratio from user prompt

def retry_image_upload(image_url, api_key, request_id=None):
    # Function to retry image uploads

def create_image_variations(image_url, user_model, n, aspect_width=None, aspect_height=None, mode=None, request_id=None):
    # Function to create variations of images
```
---
11. [routes/functions/audio_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/audio_func.py)
```python
# Functions for working with audio

def upload_audio_file(audio_file, api_key, request_id):
    # Function to upload audio files

def try_models_in_sequence(models_to_try, payload_func, api_key, request_id):
    # Function to try different models sequentially

def extract_text_from_response(response_data, request_id):
    # Function to extract transcription text from API response

def prepare_models_list(requested_model, available_models):
    # Function to prepare a list of models to try

def get_audio_from_url(audio_url, request_id):
    # Function to download audio from URL

def extract_audio_url(response_data, request_id):
    # Function to extract audio URL from API response
```
---
12. [routes/functions/file_func.py](https://github.com/chelaxian/1min-relay/blob/main/routes/functions/file_func.py)
```python
# Functions for working with files

def get_user_files(api_key, request_id=None):
    # Function to get user files from Memcached

def save_user_files(api_key, files, request_id=None):
    # Function to save user files to Memcached

def create_temp_file(file_data, suffix=".tmp", request_id=None):
    # Function to create temporary files

def upload_asset(file_data, filename, mime_type, api_key, request_id=None, file_type=None):
    # Function to upload files to 1min.ai

def get_mime_type(filename):
    # Function to determine MIME type from filename

def format_file_response(file_info, file_id=None, purpose="assistants", status="processed"):
    # Function to format file response in OpenAI format

def create_api_response(data, request_id=None):
    # Function to create HTTP response with proper headers

def find_conversation_id(response_data, request_id=None):
    # Function to find conversation ID in API response

def find_file_by_id(user_files, file_id):
    # Function to find file in user's files by ID

def create_conversation_with_files(file_ids, title, model, api_key, request_id=None):
    # Function to create a new conversation with files
```
---
13. [routes/text.py](https://github.com/chelaxian/1min-relay/blob/main/routes/text.py)
```python
# Routes for text models
@app.route("/", methods=["GET", "POST"])
def index():
    # blob/main index route

@app.route("/v1/models")
@limiter.limit("60 per minute")
def models():
    # Route to list available models

@app.route("/v1/chat/completions", methods=["POST"])
@limiter.limit("60 per minute")
def conversation():
    # Route for chat completions

@app.route("/v1/assistants", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def create_assistant():
    # Route to create assistants
```
---
14. [routes/images.py](https://github.com/chelaxian/1min-relay/blob/main/routes/images.py)
```python
# Routes for working with images
@app.route("/v1/images/generations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def generate_image():
    # Route to generate images

@app.route("/v1/images/variations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
@cross_origin()
def image_variations():
    # Route to create image variations
```
---
15. [routes/audio.py](https://github.com/chelaxian/1min-relay/blob/main/routes/audio.py)
```python
# Routes for working with audio
@app.route("/v1/audio/transcriptions", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_transcriptions():
    # Route for audio transcription

@app.route("/v1/audio/translations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def audio_translations():
    # Route for audio translation

@app.route("/v1/audio/speech", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def text_to_speech():
    # Route for text-to-speech conversion
```
---
16. [routes/files.py](https://github.com/chelaxian/1min-relay/blob/main/routes/files.py)
```python
# Routes for working with files
@app.route("/v1/files", methods=["GET", "POST", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_files():
    # Route to list or upload files

@app.route("/v1/files/<file_id>", methods=["GET", "DELETE", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file(file_id):
    # Route to get or delete specific file

@app.route("/v1/files/<file_id>/content", methods=["GET", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file_content(file_id):
    # Route to get file content
```

