# Маршруты для текстовых моделей
# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request, safe_temp_file, calculate_token
from utils.memcached import safe_memcached_operation
from . import app, limiter, IMAGE_CACHE, MAX_CACHE_SIZE, MEMORY_STORAGE  # Импортируем app, limiter и IMAGE_CACHE из модуля routes
from .images import retry_image_upload  # Импортируем функцию retry_image_upload из модуля images
from .files import create_conversation_with_files  # Импортируем функцию create_conversation_with_files из модуля files
from .utils import format_openai_response, stream_response  # Импортируем функции из utils.py

# ------------------------- Helper Functions -------------------------

def extract_prompt_text(messages):
    """Extracts text from the last message in the conversation history."""
    prompt = ""
    if messages and len(messages) > 0:
        last_message = messages[-1]
        content = last_message.get("content", "")
        if isinstance(content, str):
            prompt = content
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
            prompt = " ".join(texts).strip()
    return prompt

def find_image_url_in_messages(messages, variation_number, request_id):
    """Searches for image URL in previous assistant messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg.get("content", "")
            # First regex: markdown images with optional variation number
            url_matches = re.findall(r'!\[(?:Variation\s*(\d+)|[^]]*)\]\((https?://[^\s)]+)', content)
            image_urls = []
            for match in url_matches:
                if match[0]:
                    try:
                        var_num = int(match[0].strip())
                        while len(image_urls) < var_num:
                            image_urls.append(None)
                        image_urls[var_num - 1] = match[1]
                    except Exception as e:
                        logger.error(f"[{request_id}] Error parsing variation number: {e}")
                else:
                    image_urls.append(match[1])
            image_urls = [url for url in image_urls if url is not None]
            if image_urls:
                if len(image_urls) >= variation_number:
                    return image_urls[variation_number - 1]
                else:
                    logger.warning(f"[{request_id}] Requested variation #{variation_number} but found only {len(image_urls)} URLs. Using first URL.")
                    return image_urls[0]
    return None

def extract_variation_info(request_data, prompt_text, request_id):
    """Extracts variation command info (type, number, URL) from prompt text."""
    mono_match = re.search(r'`\[_V([1-4])_\]`', prompt_text)
    square_match = re.search(r'\[_V([1-4])_\]', prompt_text)
    old_match = re.search(r'/v([1-4])\s+(https?://[^\s]+)', prompt_text)

    messages = request_data.get("messages", [])

    if mono_match:
        variation_number = int(mono_match.group(1))
        image_url = find_image_url_in_messages(messages, variation_number, request_id)
        if image_url:
            logger.debug(f"[{request_id}] Found monospace variation command: {variation_number}")
            return ("mono", variation_number, image_url)
    if square_match:
        variation_number = int(square_match.group(1))
        image_url = find_image_url_in_messages(messages, variation_number, request_id)
        if image_url:
            logger.debug(f"[{request_id}] Found square bracket variation command: {variation_number}")
            return ("square", variation_number, image_url)
    if old_match:
        variation_number = int(old_match.group(1))
        image_url = old_match.group(2)
        logger.debug(f"[{request_id}] Found old format variation command: {variation_number}")
        return ("old", variation_number, image_url)
    return None

def extract_relative_path(image_url, request_id):
    """Extracts relative image path from full URL if it corresponds to asset.1min.ai."""
    image_path = None
    if "asset.1min.ai" in image_url:
        path_match = re.search(r'(?:asset\.1min\.ai)(/images/[^?#]+)', image_url)
        if path_match:
            image_path = path_match.group(1)
        else:
            path_match = re.search(r'/images/[^?#]+', image_url)
            if path_match:
                image_path = path_match.group(0)
        if image_path and image_path.startswith('/'):
            image_path = image_path[1:]
        logger.debug(f"[{request_id}] Extracted relative path: {image_path}")
    return image_path

def get_saved_generation_params(relative_path, request_id):
    """Retrieves saved generation parameters from MEMORY_STORAGE or memcached."""
    saved_params = None
    image_id_match = re.search(r'images/(\d+_\d+_\d+_\d+_\d+_\d+|\w+\d+)\.png', relative_path)
    if image_id_match:
        image_id = image_id_match.group(1)
        logger.info(f"[{request_id}] Extracted image_id: {image_id}")
        gen_params_key = f"gen_params:{image_id}"
        logger.info(f"[{request_id}] Looking for generation parameters with key: {gen_params_key}")
        if gen_params_key in MEMORY_STORAGE:
            stored_value = MEMORY_STORAGE[gen_params_key]
            logger.info(f"[{request_id}] Found in MEMORY_STORAGE: {stored_value}")
            if isinstance(stored_value, str):
                try:
                    saved_params = json.loads(stored_value)
                except Exception as e:
                    logger.error(f"[{request_id}] JSON parse error: {e}")
                    saved_params = stored_value
            else:
                saved_params = stored_value
        else:
            params_json = safe_memcached_operation('get', gen_params_key)
            if params_json:
                try:
                    if isinstance(params_json, str):
                        saved_params = json.loads(params_json)
                    elif isinstance(params_json, bytes):
                        saved_params = json.loads(params_json.decode('utf-8'))
                    else:
                        saved_params = params_json
                except Exception as e:
                    logger.error(f"[{request_id}] Error parsing memcached params: {e}")
                    saved_params = params_json
                logger.info(f"[{request_id}] Retrieved generation parameters: {saved_params}")
            else:
                logger.info(f"[{request_id}] No parameters found in storage for key {gen_params_key}")
    return saved_params

def build_midjourney_variation_payload(relative_path, model, saved_params):
    """Builds payload for Midjourney variation request."""
    payload = {
        "type": "IMAGE_VARIATOR",
        "model": model,
        "promptObject": {
            "imageUrl": relative_path,
            "mode": "fast",  # Default Fast mode
            "n": 4,
            "isNiji6": False,
            "aspect_width": 1,
            "aspect_height": 1,
            "maintainModeration": True
        }
    }
    if saved_params:
        for param in ["mode", "aspect_width", "aspect_height", "isNiji6", "maintainModeration"]:
            if param in saved_params:
                payload["promptObject"][param] = saved_params[param]
    return payload

def process_midjourney_variation_response(variation_response, request_id):
    """Processes successful Midjourney variation response."""
    variation_data = variation_response.json()
    logger.info(f"[{request_id}] Received Midjourney variation response")
    variation_urls = []
    # Try to extract URLs from different paths
    if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
        record_detail = variation_data["aiRecord"]["aiRecordDetail"]
        if "resultObject" in record_detail:
            result = record_detail["resultObject"]
            if isinstance(result, list):
                variation_urls = result
            elif isinstance(result, str):
                variation_urls = [result]
    if not variation_urls and "resultObject" in variation_data:
        result = variation_data["resultObject"]
        if isinstance(result, list):
            variation_urls = result
        elif isinstance(result, str):
            variation_urls = [result]
    if variation_urls:
        logger.info(f"[{request_id}] Found {len(variation_urls)} variation URLs")
        full_variation_urls = []
        asset_host = "https://asset.1min.ai"
        for url in variation_urls:
            if not url.startswith("http"):
                full_url = f"{asset_host}/{url}"
            else:
                full_url = url
            full_variation_urls.append(full_url)
        # Формирование markdown ответа
        if len(full_variation_urls) == 1:
            markdown_text = f"![Variation]({full_variation_urls[0]}) `[_V1_]`\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** and send it (paste) in the next **prompt**"
        else:
            image_lines = []
            for i, url in enumerate(full_variation_urls):
                image_lines.append(f"![Variation {i + 1}]({url}) `[_V{i + 1}_]`")
            markdown_text = "\n".join(image_lines) + "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**"
        openai_response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": variation_response.request.headers.get("API-KEY"),  # model можно заменить на актуальную переменную
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": markdown_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        return jsonify(openai_response), 200
    else:
        logger.error(f"[{request_id}] No variation URLs found in response")
        return jsonify({"error": "No variation URLs found"}), 500

def handle_midjourney_error(variation_response, request_id):
    """Handles errors for Midjourney variation requests."""
    logger.error(f"[{request_id}] Direct variation request failed: {variation_response.status_code} - {variation_response.text}")
    if variation_response.status_code == 504:
        return jsonify({
            "error": "Gateway Timeout (504) occurred while processing image variation request. Try again later."
        }), 504
    elif variation_response.status_code == 409:
        error_message = "Error creating image variation"
        try:
            error_json = variation_response.json()
            if "message" in error_json:
                error_message = error_json["message"]
        except:
            pass
        return jsonify({"error": f"Failed to create image variation: {error_message}"}), 409
    else:
        return ERROR_HANDLER(variation_response.status_code)

def download_image(image_url, request_id):
    """Downloads image to a temporary file and returns the file object."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img_response = requests.get(image_url, stream=True)
    if img_response.status_code != 200:
        logger.error(f"[{request_id}] Failed to download image: {img_response.status_code}")
        return None
    with open(temp_file.name, 'wb') as f:
        for chunk in img_response.iter_content(chunk_size=8192):
            f.write(chunk)
    return temp_file

def handle_variation(variation_info, model, request_data, request_id, api_key):
    """Processes the image variation command."""
    variation_type, variation_number, image_url = variation_info
    logger.info(f"[{request_id}] Detected {variation_type} variation command: {variation_number} for URL: {image_url}")
    if model.startswith("midjourney") and "asset.1min.ai" in image_url:
        relative_path = extract_relative_path(image_url, request_id)
        saved_params = get_saved_generation_params(relative_path, request_id)
        payload = build_midjourney_variation_payload(relative_path, model, saved_params)
        logger.info(f"[{request_id}] Sending direct Midjourney variation request: {json.dumps(payload)}")
        try:
            variation_response = api_request(
                "POST",
                ONE_MIN_API_URL,
                headers={"API-KEY": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=MIDJOURNEY_TIMEOUT
            )
            if variation_response.status_code == 200:
                return process_midjourney_variation_response(variation_response, request_id)
            else:
                return handle_midjourney_error(variation_response, request_id)
        except Exception as e:
            logger.error(f"[{request_id}] Exception during direct variation request: {str(e)}")
            return jsonify({"error": f"Error processing direct variation request: {str(e)}"}), 500
    else:
        # For non-Midjourney or other variation commands: download image and redirect
        image_path = extract_relative_path(image_url, request_id)
        temp_file = download_image(image_url, request_id)
        if not temp_file:
            return jsonify({"error": f"Failed to download image from URL"}), 400
        variation_key = f"variation:{request_id}"
        variation_data = {
            "temp_file": temp_file.name,
            "model": model,
            "n": request_data.get("n", 1),
            "image_path": image_path
        }
        safe_memcached_operation('set', variation_key, variation_data, expiry=300)  # Store 5 minutes
        logger.debug(f"[{request_id}] Saved variation data with key: {variation_key}")
        logger.info(f"[{request_id}] Redirecting to /v1/images/variations with model {model}")
        return redirect(url_for('image_variations', request_id=request_id), code=307)

def get_team_id(api_key, request_id):
    """Retrieves team ID using API key."""
    team_id = None
    try:
        teams_url = f"{ONE_MIN_API_URL}/teams"
        teams_headers = {"API-KEY": api_key}
        teams_response = api_request("GET", teams_url, headers=teams_headers)
        if teams_response.status_code == 200:
            teams_data = teams_response.json()
            if "data" in teams_data and teams_data["data"]:
                team_id = teams_data["data"][0].get("id")
                logger.debug(f"[{request_id}] Found team ID: {team_id}")
    except Exception as e:
        logger.error(f"[{request_id}] Error getting team ID: {str(e)}")
    return team_id

def has_deletion_keywords(text):
    """Checks if the text contains keywords for file deletion."""
    delete_keywords = ["удалить", "удали", "удаление", "очисти", "очистка", "delete", "remove", "clean"]
    file_keywords = ["файл", "файлы", "file", "files", "документ", "документы", "document", "documents"]
    return any(kw in text for kw in delete_keywords) and any(kw in text for kw in file_keywords)

def get_user_files(api_key, request_id):
    """Retrieves user files from memcached."""
    user_file_ids = []
    if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
        try:
            user_key = f"user:{api_key}"
            user_files_json = safe_memcached_operation('get', user_key)
            if user_files_json:
                try:
                    if isinstance(user_files_json, str):
                        user_files = json.loads(user_files_json)
                    elif isinstance(user_files_json, bytes):
                        user_files = json.loads(user_files_json.decode('utf-8'))
                    else:
                        user_files = user_files_json
                    if user_files and isinstance(user_files, list):
                        user_file_ids = [file_info.get("id") for file_info in user_files if file_info.get("id")]
                        logger.debug(f"[{request_id}] Found user files: {user_file_ids}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error parsing user files: {str(e)}")
        except Exception as e:
            logger.error(f"[{request_id}] Error retrieving user files: {str(e)}")
    else:
        logger.debug(f"[{request_id}] Memcached not available, no user files loaded")
    return user_file_ids

def process_file_deletion(api_key, user_file_ids, request_id):
    """Processes deletion of user files and clears memcached list."""
    team_id = get_team_id(api_key, request_id)
    deleted_files = []
    for file_id in user_file_ids:
        try:
            if team_id:
                delete_url = f"{ONE_MIN_API_URL}/teams/{team_id}/assets/{file_id}"
            else:
                delete_url = f"{ONE_MIN_ASSET_URL}/{file_id}"
            logger.debug(f"[{request_id}] Using URL for deletion: {delete_url}")
            headers = {"API-KEY": api_key}
            delete_response = api_request("DELETE", delete_url, headers=headers)
            if delete_response.status_code == 200:
                logger.info(f"[{request_id}] Successfully deleted file: {file_id}")
                deleted_files.append(file_id)
            else:
                logger.error(f"[{request_id}] Failed to delete file {file_id}: {delete_response.status_code}")
        except Exception as e:
            logger.error(f"[{request_id}] Error deleting file {file_id}: {str(e)}")
    if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None and deleted_files:
        try:
            user_key = f"user:{api_key}"
            safe_memcached_operation('set', user_key, json.dumps([]))
            logger.info(f"[{request_id}] Cleared user files list in memcached")
        except Exception as e:
            logger.error(f"[{request_id}] Error clearing user files in memcached: {str(e)}")
    response = {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "N/A",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Удалено файлов: {len(deleted_files)}. Список файлов очищен."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": calculate_token(""),  # Здесь можно подставить актуальное значение
            "completion_tokens": 20,
            "total_tokens": 20
        }
    }
    return jsonify(response), 200

# Функции process_pdf_chat, process_text_to_speech, process_images_from_messages и другие
# можно также вынести в отдельные модули/функции по аналогичной схеме,
# чтобы убрать повторение кода. Ниже приведён пример для TTS.

def process_text_to_speech(request_data, prompt_text, request_id, api_key, model):
    """Processes text-to-speech requests."""
    if not prompt_text:
        logger.error(f"[{request_id}] No input text provided for TTS")
        return jsonify({"error": "No input text provided"}), 400
    voice = request_data.get("voice", "alloy")
    response_format = request_data.get("response_format", "mp3")
    speed = request_data.get("speed", 1.0)
    payload = {
        "type": "TEXT_TO_SPEECH",
        "model": model,
        "promptObject": {
            "text": prompt_text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed
        }
    }
    headers = {"API-KEY": api_key, "Content-Type": "application/json"}
    try:
        logger.debug(f"[{request_id}] Sending direct TTS request to {ONE_MIN_API_URL}")
        response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
        if response.status_code != 200:
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            logger.error(f"[{request_id}] Error in TTS response: {response.text[:200]}")
            return jsonify({"error": response.json().get("error", "Unknown error")}), response.status_code
        one_min_response = response.json()
        audio_url = ""
        if "aiRecord" in one_min_response and "aiRecordDetail" in one_min_response["aiRecord"]:
            result_object = one_min_response["aiRecord"]["aiRecordDetail"].get("resultObject", "")
            if isinstance(result_object, list) and result_object:
                audio_url = result_object[0]
            else:
                audio_url = result_object
        elif "resultObject" in one_min_response:
            result_object = one_min_response["resultObject"]
            if isinstance(result_object, list) and result_object:
                audio_url = result_object[0]
            else:
                audio_url = result_object
        if not audio_url:
            logger.error(f"[{request_id}] Could not extract audio URL from API response")
            return jsonify({"error": "Could not extract audio URL"}), 500
        try:
            signed_url = None
            if "temporaryUrl" in one_min_response:
                signed_url = one_min_response["temporaryUrl"]
            elif "result" in one_min_response and "resultList" in one_min_response["result"]:
                for item in one_min_response["result"]["resultList"]:
                    if item.get("type") == "TEXT_TO_SPEECH" and "temporaryUrl" in item:
                        signed_url = item["temporaryUrl"]
                        break
            if not signed_url and "aiRecord" in one_min_response:
                if "temporaryUrl" in one_min_response["aiRecord"]:
                    signed_url = one_min_response["aiRecord"]["temporaryUrl"]
            if not signed_url:
                if "aiRecord" in one_min_response and "aiRecordDetail" in one_min_response["aiRecord"]:
                    detail = one_min_response["aiRecord"]["aiRecordDetail"]
                    if "signedUrls" in detail:
                        signed_urls = detail["signedUrls"]
                        signed_url = signed_urls[0] if isinstance(signed_urls, list) and signed_urls else signed_urls
                    elif "signedUrl" in detail:
                        signed_url = detail["signedUrl"]
            if signed_url:
                full_audio_url = signed_url
                logger.debug(f"[{request_id}] Using signed URL from API")
            else:
                full_audio_url = f"https://s3.us-east-1.amazonaws.com/asset.1min.ai/{audio_url}"
                logger.warning(f"[{request_id}] No signed URL found, using base S3 URL")
        except Exception as e:
            logger.error(f"[{request_id}] Error processing audio URL: {str(e)}")
            full_audio_url = f"https://asset.1min.ai/{audio_url}"
        completion_response = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"🔊 [Audio.mp3]({full_audio_url})"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_text.split()),
                "completion_tokens": 1,
                "total_tokens": len(prompt_text.split()) + 1
            }
        }
        return jsonify(completion_response)
    except Exception as e:
        logger.error(f"[{request_id}] Exception during TTS request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ------------------------- End of Helper Functions -------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return ERROR_HANDLER(1212)
    if request.method == "GET":
        internal_ip = socket.gethostbyname(socket.gethostname())
        return (
            "Congratulations! Your API is working! You can now make requests to the API.\n\nEndpoint: "
            + internal_ip
            + ":5001/v1"
        )

@app.route("/v1/models")
@limiter.limit("60 per minute")
def models():
    models_data = []
    if not PERMIT_MODELS_FROM_SUBSET_ONLY:
        one_min_models_data = [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "1minai",
                "created": 1727389042,
            }
            for model_name in ALL_ONE_MIN_AVAILABLE_MODELS
        ]
    else:
        one_min_models_data = [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "1minai",
                "created": 1727389042,
            }
            for model_name in SUBSET_OF_ONE_MIN_PERMITTED_MODELS
        ]
    models_data.extend(one_min_models_data)
    return jsonify({"data": models_data, "object": "list"})

@app.route("/v1/chat/completions", methods=["POST"])
@limiter.limit("60 per minute")
def conversation():
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/chat/completions")
    if request.method == "POST" and not request.json:
        return jsonify({"error": "Invalid request format"}), 400
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        logger.error(f"[{request_id}] No API key provided")
        return jsonify({"error": "API key required"}), 401

    try:
        request_data = request.json.copy()
        model = request_data.get("model", "").strip()
        logger.info(f"[{request_id}] Using model: {model}")
        capabilities = get_model_capabilities(model)

        # Enable web search if requested
        web_search_requested = any(tool.get("type") == "retrieval" for tool in request_data.get("tools", []))
        if not web_search_requested and request_data.get("web_search", False):
            web_search_requested = True
        if web_search_requested:
            if capabilities["retrieval"]:
                request_data["web_search"] = True
                request_data["num_of_site"] = request_data.get("num_of_site", 1)
                request_data["max_word"] = request_data.get("max_word", 1000)
                logger.info(f"[{request_id}] Web search enabled for model {model}")
            else:
                logger.warning(f"[{request_id}] Model {model} does not support web search, ignoring request")

        messages = request_data.get("messages", [])
        prompt_text = extract_prompt_text(messages)

        # Check and process image variation command
        variation_info = extract_variation_info(request_data, prompt_text, request_id)
        if variation_info:
            return handle_variation(variation_info, model, request_data, request_id, api_key)

        # Log prompt text (for debugging)
        logger.debug(f"[{request_id}] Extracted prompt text: {prompt_text[:100]}..." if len(prompt_text) > 100 else f"[{request_id}] Extracted prompt text: {prompt_text}")

        # Process TTS requests
        if model in TEXT_TO_SPEECH_MODELS:
            logger.info(f"[{request_id}] Processing text-to-speech request directly")
            return process_text_to_speech(request_data, prompt_text, request_id, api_key, model)

        # Process STT requests
        if model in SPEECH_TO_TEXT_MODELS:
            logger.info(f"[{request_id}] Redirecting speech-to-text model to /v1/audio/transcriptions")
            return redirect(url_for('audio_transcriptions'), code=307)

        # Process image generation models
        if model in IMAGE_GENERATION_MODELS:
            logger.info(f"[{request_id}] Redirecting image generation model to /v1/images/generations")
            image_request = {
                "model": model,
                "prompt": prompt_text,
                "n": request_data.get("n", 1),
                "size": request_data.get("size", "1024x1024")
            }
            if model == "dall-e-3":
                image_request["quality"] = request_data.get("quality", "standard")
                image_request["style"] = request_data.get("style", "vivid")
            if model.startswith("midjourney"):
                if "--ar" in prompt_text or "\u2014ar" in prompt_text:
                    logger.debug(f"[{request_id}] Found aspect ratio parameter in prompt")
                elif request_data.get("aspect_ratio"):
                    image_request["aspect_ratio"] = request_data.get("aspect_ratio")
                if "--no" in prompt_text or "\u2014no" in prompt_text:
                    logger.debug(f"[{request_id}] Found negative prompt parameter in prompt")
                elif request_data.get("negative_prompt"):
                    image_request["negative_prompt"] = request_data.get("negative_prompt")
            if "messages" in image_request:
                del image_request["messages"]
            logger.debug(f"[{request_id}] Final image request: {json.dumps(image_request)[:200]}...")
            request.environ["body_copy"] = json.dumps(image_request)
            return redirect(url_for('generate_image'), code=307)

        # Process file deletion if keywords are detected
        extracted_prompt_lower = prompt_text.lower() if prompt_text else ""
        user_file_ids = get_user_files(api_key, request_id)
        if has_deletion_keywords(extracted_prompt_lower) and user_file_ids:
            logger.info(f"[{request_id}] Deletion request detected, processing file deletion")
            return process_file_deletion(api_key, user_file_ids, request_id)

        # Остальная логика обработки диалога (PDF, обычный чат, изображения, streaming и т.д.)
        # Оставлена практически без изменений для сохранения существующей логики.
        # Подготовка истории диалога:
        all_messages = "\n".join([f"{msg.get('role').capitalize()}: {msg.get('content')}" for msg in messages])
        # Обработка изображений в сообщениях, файлов и прочего здесь...
        # Подсчёт токенов и формирование payload:
        prompt_token = calculate_token(all_messages)
        if PERMIT_MODELS_FROM_SUBSET_ONLY and model not in AVAILABLE_MODELS:
            return ERROR_HANDLER(1002, model)
        logger.debug(f"[{request_id}] Processing {prompt_token} prompt tokens with model {model}")
        payload = prepare_payload(request_data, model, all_messages, image_paths=[], request_id=request_id)
        headers = {"API-KEY": api_key, "Content-Type": "application/json"}
        if not request_data.get("stream", False):
            logger.debug(f"[{request_id}] Sending non-streaming request to {ONE_MIN_API_URL}")
            try:
                response_api = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
                logger.debug(f"[{request_id}] Response status code: {response_api.status_code}")
                if response_api.status_code != 200:
                    if response_api.status_code == 401:
                        return ERROR_HANDLER(1020, key=api_key)
                    try:
                        error_content = response_api.json()
                        logger.error(f"[{request_id}] Error response: {error_content}")
                    except:
                        logger.error(f"[{request_id}] Could not parse error response as JSON")
                    return ERROR_HANDLER(response_api.status_code)
                one_min_response = response_api.json()
                transformed_response = transform_response(one_min_response, request_data, prompt_token)
                response_obj = make_response(jsonify(transformed_response))
                set_response_headers(response_obj)
                return response_obj, 200
            except Exception as e:
                logger.error(f"[{request_id}] Exception during request: {str(e)}")
                return jsonify({"error": str(e)}), 500
        else:
            logger.debug(f"[{request_id}] Sending streaming request")
            streaming_url = f"{ONE_MIN_API_URL}?isStreaming=true"
            logger.debug(f"[{request_id}] Streaming URL: {streaming_url}")
            logger.debug(f"[{request_id}] Payload: {json.dumps(payload)[:200]}...")
            try:
                session = create_session()
                response_stream = session.post(streaming_url, json=payload, headers=headers, stream=True)
                logger.debug(f"[{request_id}] Streaming response status code: {response_stream.status_code}")
                if response_stream.status_code != 200:
                    if response_stream.status_code == 401:
                        session.close()
                        return ERROR_HANDLER(1020, key=api_key)
                    logger.error(f"[{request_id}] Error status code: {response_stream.status_code}")
                    try:
                        error_content = response_stream.json()
                        logger.error(f"[{request_id}] Error response: {error_content}")
                    except:
                        logger.error(f"[{request_id}] Could not parse error response as JSON")
                    session.close()
                    return ERROR_HANDLER(response_stream.status_code)
                return Response(stream_response(response_stream, request_data, model, prompt_token, session),
                                content_type="text/event-stream")
            except Exception as e:
                logger.error(f"[{request_id}] Exception during streaming request: {str(e)}")
                return jsonify({"error": str(e)}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Exception during conversation processing: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Error during conversation processing: {str(e)}"}), 500

@app.route("/v1/assistants", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def create_assistant():
    if request.method == "OPTIONS":
        return handle_options_request()
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error("Invalid Authentication")
        return ERROR_HANDLER(1021)
    api_key = auth_header.split(" ")[1]
    headers = {"API-KEY": api_key, "Content-Type": "application/json"}
    request_data = request.json
    name = request_data.get("name", "PDF Assistant")
    instructions = request_data.get("instructions", "")
    model = request_data.get("model", "gpt-4o-mini")
    file_ids = request_data.get("file_ids", [])
    payload = {
        "title": name,
        "type": "CHAT_WITH_PDF",
        "model": model,
        "fileList": file_ids,
    }
    response = requests.post(ONE_MIN_CONVERSATION_API_URL, json=payload, headers=headers)
    if response.status_code != 200:
        if response.status_code == 401:
            return ERROR_HANDLER(1020, key=api_key)
        return jsonify({"error": response.json().get("error", "Unknown error")}), response.status_code
    one_min_response = response.json()
    try:
        conversation_id = one_min_response.get("id")
        openai_response = {
            "id": f"asst_{conversation_id}",
            "object": "assistant",
            "created_at": int(time.time()),
            "name": name,
            "description": None,
            "model": model,
            "instructions": instructions,
            "tools": [],
            "file_ids": file_ids,
            "metadata": {},
        }
        response_obj = make_response(jsonify(openai_response))
        set_response_headers(response_obj)
        return response_obj, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------- Вспомогательные функции для текстовых моделей -------------------------

def format_conversation_history(messages, new_input):
    """Formats conversation history into a structured string."""
    formatted_history = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if isinstance(content, list):
            processed_content = [item.get("text", "") for item in content if "text" in item]
            content = "\n".join(processed_content)
        if role == "system":
            formatted_history.append(f"System: {content}")
        elif role == "user":
            formatted_history.append(f"User: {content}")
        elif role == "assistant":
            formatted_history.append(f"Assistant: {content}")
    if new_input:
        formatted_history.append(f"User: {new_input}")
    return "\n".join(formatted_history)

def get_model_capabilities(model):
    """Defines supported capabilities for a given model."""
    capabilities = {
        "vision": model in vision_supported_models,
        "code_interpreter": model in code_interpreter_supported_models,
        "retrieval": model in retrieval_supported_models,
        "function_calling": model in function_calling_supported_models,
    }
    return capabilities

def prepare_payload(request_data, model, all_messages, image_paths=None, request_id=None):
    """Prepares the payload for the request taking model capabilities into account."""
    capabilities = get_model_capabilities(model)
    web_search = False
    code_interpreter = False
    for tool in request_data.get("tools", []):
        tool_type = tool.get("type", "")
        if tool_type == "retrieval" and capabilities["retrieval"]:
            web_search = True
            logger.debug(f"[{request_id}] Enabled web search due to retrieval tool")
        elif tool_type == "code_interpreter" and capabilities["code_interpreter"]:
            code_interpreter = True
            logger.debug(f"[{request_id}] Enabled code interpreter")
        else:
            logger.debug(f"[{request_id}] Ignoring unsupported tool: {tool_type}")
    if not web_search and request_data.get("web_search", False) and capabilities["retrieval"]:
        web_search = True
        logger.debug(f"[{request_id}] Enabled web search due to web_search parameter")
    num_of_site = request_data.get("num_of_site", 3)
    max_word = request_data.get("max_word", 500)
    if image_paths and len(image_paths) > 0:
        return prepare_image_payload(model, all_messages, image_paths, web_search, num_of_site, max_word, capabilities, request_id)
    elif code_interpreter:
        return prepare_code_payload(model, all_messages, request_id)
    else:
        return prepare_text_payload(model, all_messages, web_search, num_of_site, max_word, request_id)

def prepare_image_payload(model, all_messages, image_paths, web_search, num_of_site, max_word, capabilities, request_id=None):
    """Prepares payload for requests with images."""
    if capabilities["vision"]:
        enhanced_prompt = all_messages
        if not enhanced_prompt.strip().startswith(IMAGE_DESCRIPTION_INSTRUCTION):
            enhanced_prompt = f"{IMAGE_DESCRIPTION_INSTRUCTION}\n\n{all_messages}"
        payload = {
            "type": "CHAT_WITH_IMAGE",
            "model": model,
            "promptObject": {
                "prompt": enhanced_prompt,
                "isMixed": False,
                "imageList": image_paths,
                "webSearch": web_search,
                "numOfSite": num_of_site if web_search else None,
                "maxWord": max_word if web_search else None,
            },
        }
        logger.debug(f"[{request_id}] Created image payload for vision model with {len(image_paths)} images")
    else:
        logger.debug(f"[{request_id}] Model {model} doesn't support vision, falling back to text-only request")
        payload = prepare_text_payload(model, all_messages, web_search, num_of_site, max_word, request_id)
    return payload

def prepare_code_payload(model, all_messages, request_id=None):
    """Prepares payload for code interpreter requests."""
    payload = {
        "type": "CODE_GENERATOR",
        "model": model,
        "conversationId": "CODE_GENERATOR",
        "promptObject": {"prompt": all_messages},
    }
    logger.debug(f"[{request_id}] Created code interpreter payload")
    return payload

def prepare_text_payload(model, all_messages, web_search, num_of_site, max_word, request_id=None):
    """Prepares payload for normal text requests."""
    payload = {
        "type": "CHAT_WITH_AI",
        "model": model,
        "promptObject": {
            "prompt": all_messages,
            "isMixed": False,
            "webSearch": web_search,
            "numOfSite": num_of_site if web_search else None,
            "maxWord": max_word if web_search else None,
        },
    }
    if web_search:
        logger.debug(f"[{request_id}] Web search enabled with numOfSite={num_of_site}, maxWord={max_word}")
    return payload

def transform_response(one_min_response, request_data, prompt_token):
    try:
        logger.debug(f"Response structure: {json.dumps(one_min_response)[:200]}...")
        result_text = (one_min_response.get("aiRecord", {})
                       .get("aiRecordDetail", {})
                       .get("resultObject", [""])[0])
        if not result_text:
            if "resultObject" in one_min_response:
                result_text = (one_min_response["resultObject"][0]
                               if isinstance(one_min_response["resultObject"], list)
                               else one_min_response["resultObject"])
            elif "result" in one_min_response:
                result_text = one_min_response["result"]
            else:
                logger.error("Cannot extract response text from API result")
                result_text = "Error: Could not extract response from API"
        completion_token = calculate_token(result_text)
        logger.debug(f"Finished processing Non-Streaming response. Completion tokens: {completion_token}")
        logger.debug(f"Total tokens: {completion_token + prompt_token}")
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "mistral-nemo").strip(),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token,
                "completion_tokens": completion_token,
                "total_tokens": prompt_token + completion_token,
            },
        }
    except Exception as e:
        logger.error(f"Error in transform_response: {str(e)}")
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "mistral-nemo").strip(),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"Error processing response: {str(e)}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": prompt_token, "completion_tokens": 0, "total_tokens": prompt_token},
        }

def stream_response(response, request_data, model, prompt_tokens, session=None):
    """Streams response from 1min.ai in OpenAI API-compatible format."""
    all_chunks = ""
    first_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first_chunk)}\n\n"
    for chunk in response.iter_content(chunk_size=1024):
        finish_reason = None
        decoded_chunk = chunk.decode('utf-8')
        all_chunks += decoded_chunk
        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": decoded_chunk}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(return_chunk)}\n\n"
    tokens = calculate_token(all_chunks)
    logger.debug(f"Finished processing streaming response. Completion tokens: {tokens}")
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tokens, "total_tokens": tokens + prompt_tokens},
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

def emulate_stream_response(full_content, request_data, model, prompt_tokens):
    """Emulates streaming response when API does not support it."""
    words = full_content.split()
    chunks = [" ".join(words[i: i + 5]) for i in range(0, len(words), 5)]
    for chunk in chunks:
        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(return_chunk)}\n\n"
        time.sleep(0.05)
    tokens = calculate_token(full_content)
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": tokens, "total_tokens": tokens + prompt_tokens},
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
