# routes/images.py

# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import (
    ERROR_HANDLER, 
    handle_options_request, 
    set_response_headers, 
    create_session, 
    api_request, 
    safe_temp_file, 
    calculate_token
)
from utils.memcached import safe_memcached_operation
from routes.functions.shared_func import validate_auth, handle_api_error, format_image_response, get_full_url, extract_image_urls
from routes.functions.img_func import build_generation_payload, parse_aspect_ratio, create_image_variations, retry_image_upload
from . import app, limiter, MEMORY_STORAGE  # Импортируем app, limiter и MEMORY_STORAGE из модуля routes

# ----------------------- Endpoints -----------------------

@app.route("/v1/images/generations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def generate_image():
    if request.method == "OPTIONS":
        return handle_options_request()
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: /v1/images/generations")

    # Validate authentication
    api_key, error = validate_auth(request, request_id)
    if error:
        return error
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)
    api_key = auth_header.split(" ")[1]
    headers = {"API-KEY": api_key, "Content-Type": "application/json"}

    if not request.is_json:
        logger.error(f"[{request_id}] Request content-type is not application/json")
        return jsonify({"error": "Content-type must be application/json"}), 400
    request_data = request.get_json()

    model = request_data.get("model", "dall-e-3").strip()
    prompt = request_data.get("prompt", "").strip()

    # Если запрос пришёл с chat/completions – используем только последний запрос пользователя
    if request.environ.get("HTTP_REFERER") and "chat/completions" in request.environ.get("HTTP_REFERER"):
        logger.debug(f"[{request_id}] Request came from chat completions, isolating the prompt")

    negative_prompt = None
    no_match = re.search(r'(--|\u2014)no\s+(.*?)(?=(--|\u2014)|$)', prompt)
    if no_match:
        negative_prompt = no_match.group(2).strip()
        prompt = re.sub(r'(--|\u2014)no\s+.*?(?=(--|\u2014)|$)', '', prompt).strip()

    prompt, aspect_ratio, size, ar_error, mode = parse_aspect_ratio(prompt, model, request_data, request_id)
    if ar_error:
        return jsonify({"error": ar_error}), 400

    if not prompt:
        messages = request_data.get("messages", [])
        if messages:
            last_message = messages[-1]
            content = last_message.get("content", "")
            if isinstance(content, str):
                prompt = content
            elif isinstance(content, list):
                prompt = " ".join([item.get("text", "") for item in content if isinstance(item, dict)])
            negative_prompt = None
            no_match = re.search(r'(--|\u2014)no\s+(.*?)(?=(--|\u2014)|$)', prompt)
            if no_match:
                negative_prompt = no_match.group(2).strip()
            prompt, aspect_ratio, size, ar_error, mode = parse_aspect_ratio(prompt, model, request_data, request_id)
            if ar_error:
                return jsonify({"error": ar_error}), 400
        if not prompt:
            logger.error(f"[{request_id}] No prompt provided")
            return jsonify({"error": "No prompt provided"}), 400

    logger.info(f"[{request_id}] Using model: {model}, prompt: '{prompt}'")

    try:
        api_url = f"{ONE_MIN_API_URL}"
        timeout = MIDJOURNEY_TIMEOUT if model in ["midjourney", "midjourney_6_1"] else DEFAULT_TIMEOUT

        payload, payload_error = build_generation_payload(model, prompt, request_data, negative_prompt, aspect_ratio, size, mode, request_id)
        if payload_error:
            return payload_error

        logger.debug(f"[{request_id}] Sending request to API: {api_url}")
        logger.debug(f"[{request_id}] Payload: {json.dumps(payload)[:500]}")

        response = api_request("POST", api_url, headers=headers, json=payload, timeout=timeout, stream=False)
        logger.debug(f"[{request_id}] Response status code: {response.status_code}")

        if response.status_code == 200:
            api_response = response.json()
        else:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = error_data["error"]
            except:
                pass
            if response.status_code == 401:
                return ERROR_HANDLER(1020, key=api_key)
            return jsonify({"error": error_msg}), response.status_code

        image_urls = extract_image_urls(api_response, request_id)
        if not image_urls:
            return jsonify({"error": "Could not extract image URLs from API response"}), 500

        logger.debug(f"[{request_id}] Successfully generated {len(image_urls)} images")
        # Сохраняем параметры генерации для Midjourney
        if model in ["midjourney", "midjourney_6_1"]:
            for url in image_urls:
                if url:
                    image_id_match = re.search(r'images/(\d+_\d+_\d+_\d+_\d+_\d+|\w+\d+)\.png', url)
                    if image_id_match:
                        image_id = image_id_match.group(1)
                        logger.info(f"[{request_id}] Extracted image_id from URL: {image_id}")
                        gen_params = {
                            "mode": payload["promptObject"].get("mode", "fast"),
                            "aspect_width": payload["promptObject"].get("aspect_width", 1),
                            "aspect_height": payload["promptObject"].get("aspect_height", 1),
                            "isNiji6": payload["promptObject"].get("isNiji6", False),
                            "maintainModeration": payload["promptObject"].get("maintainModeration", True)
                        }
                        gen_params_key = f"gen_params:{image_id}"
                        safe_memcached_operation('set', gen_params_key, gen_params, expiry=3600*24*7)
                        logger.info(f"[{request_id}] Saved generation parameters for image {image_id}: {gen_params}")

        full_image_urls = [get_full_url(url) for url in image_urls if url]

        openai_data = []
        for i, url in enumerate(full_image_urls):
            if model in IMAGE_VARIATION_MODELS:
                openai_data.append({
                    "url": url,
                    "revised_prompt": prompt,
                    "variation_commands": {"variation": f"/v{i + 1} {url}"}
                })
            else:
                openai_data.append({"url": url, "revised_prompt": prompt})

        markdown_text = ""
        if len(full_image_urls) == 1:
            markdown_text = f"![Image]({full_image_urls[0]}) `[_V1_]`"
        else:
            markdown_text = "\n".join([f"![Image {i+1}]({url}) `[_V{i+1}_]`" for i, url in enumerate(full_image_urls)])
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]**" \
                         " - **[_V4_]** and send it (paste) in the next **prompt**"

        openai_response = {
            "created": int(time.time()),
            "data": openai_data,
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": markdown_text,
                    "structured_output": {"type": "image", "image_urls": full_image_urls}
                },
                "index": 0,
                "finish_reason": "stop"
            }]
        }
        response_obj = make_response(jsonify(openai_response))
        set_response_headers(response_obj)
        return response_obj, 200
    except Exception as e:
        logger.error(f"[{request_id}] Error during image generation: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/images/variations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
@cross_origin()
def image_variations():
    """
    Process the image variation request. Supports both JSON (with base64 image) and form-data (with image file).
    """
    if request.method == "OPTIONS":
        return handle_options_request()
    request_id = str(uuid.uuid4())
    logger.debug(f"[{request_id}] Processing image variation request")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)
    api_key = auth_header.split(" ")[1]

    # Если запрос перенаправлен (передан request_id)
    if 'request_id' in request.args:
        redirect_request_id = request.args.get('request_id')
        variation_key = f"variation:{redirect_request_id}"
        logger.info(f"[{request_id}] Looking for variation data with key: {variation_key}")
        variation_data_json = safe_memcached_operation('get', variation_key)
        if variation_data_json:
            try:
                if isinstance(variation_data_json, str):
                    variation_data = json.loads(variation_data_json)
                elif isinstance(variation_data_json, bytes):
                    variation_data = json.loads(variation_data_json.decode('utf-8'))
                else:
                    variation_data = variation_data_json
                temp_file_path = variation_data.get("temp_file")
                model = variation_data.get("model")
                n = variation_data.get("n", 1)
                image_path = variation_data.get("image_path")
                logger.debug(f"[{request_id}] Retrieved variation data: model={model}, n={n}, temp_file={temp_file_path}")
                if image_path:
                    logger.debug(f"[{request_id}] Retrieved image path: {image_path}")
                if os.path.exists(temp_file_path):
                    with open(temp_file_path, 'rb') as f:
                        file_data = f.read()
                    logger.info(f"[{request_id}] Read temporary file, size: {len(file_data)} bytes")
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    temp_file.write(file_data)
                    temp_file.close()
                    from io import BytesIO
                    file_data_io = BytesIO(file_data)
                    from werkzeug.datastructures import FileStorage
                    file_storage = FileStorage(stream=file_data_io, filename="variation.png", content_type="image/png")
                    request.files = {"image": file_storage}
                    form_data = [("model", model), ("n", str(n))]
                    if image_path:
                        form_data.append(("image_path", image_path))
                        logger.info(f"[{request_id}] Added image_path to form_data: {image_path}")
                    request.form = MultiDict(form_data)
                    logger.info(f"[{request_id}] Using file from memcached for image variations")
                    try:
                        os.unlink(temp_file_path)
                        logger.debug(f"[{request_id}] Deleted original temporary file: {temp_file_path}")
                    except Exception as e:
                        logger.warning(f"[{request_id}] Failed to delete temporary file: {str(e)}")
                else:
                    logger.error(f"[{request_id}] Temporary file not found: {temp_file_path}")
                    return jsonify({"error": "Image file not found"}), 400
            except Exception as e:
                logger.error(f"[{request_id}] Error processing variation data: {str(e)}")
                return jsonify({"error": f"Error processing variation request: {str(e)}"}), 500

    if "image" not in request.files:
        logger.error(f"[{request_id}] No image file provided")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    original_model = request.form.get("model", "dall-e-2").strip()
    n = int(request.form.get("n", 1))
    size = request.form.get("size", "1024x1024")
    prompt_text = request.form.get("prompt", "")
    relative_image_path = request.form.get("image_path")
    if relative_image_path:
        logger.debug(f"[{request_id}] Using relative image path: {relative_image_path}")
    logger.debug(f"[{request_id}] Original model requested: {original_model} for image variations")

    fallback_models = ["midjourney_6_1", "midjourney", "clipdrop", "dall-e-2"]
    if original_model in IMAGE_VARIATION_MODELS:
        models_to_try = [original_model] + [m for m in fallback_models if m != original_model]
    else:
        logger.warning(f"[{request_id}] Model {original_model} does not support image variations. Using fallback models.")
        models_to_try = fallback_models

    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image_file.save(temp_file.name)
        temp_file.close()
    except Exception as e:
        logger.error(f"[{request_id}] Failed to save temporary file: {str(e)}")
        return jsonify({"error": "Failed to process image file"}), 500

    session = create_session()
    headers = {"API-KEY": api_key}

    aspect_width, aspect_height = 1, 1
    if "--ar" in prompt_text:
        ar_match = re.search(r'--ar\s+(\d+):(\d+)', prompt_text)
        if ar_match:
            aspect_width = int(ar_match.group(1))
            aspect_height = int(ar_match.group(2))
            logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_width}:{aspect_height}")

    variation_urls = []
    current_model = None

    # Перебор моделей для вариаций
    for model in models_to_try:
        current_model = model
        logger.info(f"[{request_id}] Trying model: {model} for image variations")
        try:
            # Определяем MIME-type и расширение
            content_type = "image/png"
            if "content-type" in image_response.headers:
                content_type = image_response.headers["content-type"]
            elif image_url.lower().endswith(".webp"):
                content_type = "image/webp"
            elif image_url.lower().endswith(".jpg") or image_url.lower().endswith(".jpeg"):
                content_type = "image/jpeg"
            elif image_url.lower().endswith(".gif"):
                content_type = "image/gif"
            ext = "png"
            if "webp" in content_type:
                ext = "webp"
            elif "jpeg" in content_type or "jpg" in content_type:
                ext = "jpg"
            elif "gif" in content_type:
                ext = "gif"
            logger.debug(f"[{request_id}] Detected image type: {content_type}, extension: {ext}")

            # Загружаем изображение на сервер
            with open(temp_file.name, 'rb') as img_file:
                files = {"asset": (f"variation.{ext}", img_file, content_type)}
                asset_response = session.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
                logger.debug(f"[{request_id}] Image upload response status: {asset_response.status_code}")
                if asset_response.status_code != 200:
                    logger.error(f"[{request_id}] Image upload failed: {asset_response.status_code}")
                    continue
                asset_data = asset_response.json()
                logger.debug(f"[{request_id}] Asset upload response: {asset_data}")
                image_path = None
                if "fileContent" in asset_data and "path" in asset_data["fileContent"]:
                    image_path = asset_data["fileContent"]["path"]
                    if image_path.startswith('/'):
                        image_path = image_path[1:]
                    logger.debug(f"[{request_id}] Using relative path for variation: {image_path}")
                else:
                    logger.error(f"[{request_id}] Could not extract image path from upload response")
                    continue

            relative_image_url = image_path
            if relative_image_url and relative_image_url.startswith('/'):
                relative_image_url = relative_image_url[1:]

            if model.startswith("midjourney"):
                payload = {
                    "type": "IMAGE_VARIATOR",
                    "model": model,
                    "promptObject": {
                        "imageUrl": relative_image_url,
                        "mode": request.form.get("mode", "fast"),
                        "n": 4,
                        "isNiji6": False,
                        "aspect_width": aspect_width,
                        "aspect_height": aspect_height,
                        "maintainModeration": True
                    }
                }
            elif model == "dall-e-2":
                payload = {
                    "type": "IMAGE_VARIATOR",
                    "model": "dall-e-2",
                    "promptObject": {
                        "imageUrl": relative_image_url,
                        "n": 1,
                        "size": "1024x1024"
                    }
                }
                logger.info(f"[{request_id}] DALL-E 2 variation payload: {json.dumps(payload, indent=2)}")
                variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=MIDJOURNEY_TIMEOUT)
                if variation_response.status_code != 200:
                    logger.error(f"[{request_id}] DALL-E 2 variation failed: {variation_response.status_code}, {variation_response.text}")
                    continue
                variation_data = variation_response.json()
                if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                    result_object = variation_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
                    if isinstance(result_object, list):
                        variation_urls.extend(result_object)
                    elif isinstance(result_object, str):
                        variation_urls.append(result_object)
                elif "resultObject" in variation_data:
                    result_object = variation_data["resultObject"]
                    if isinstance(result_object, list):
                        variation_urls.extend(result_object)
                    elif isinstance(result_object, str):
                        variation_urls.append(result_object)
                if variation_urls:
                    logger.info(f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2")
                    break
                else:
                    logger.warning(f"[{request_id}] No variation URLs found in DALL-E 2 response")
            elif model == "clipdrop":
                payload = {
                    "type": "IMAGE_VARIATOR",
                    "model": "clipdrop",
                    "promptObject": {
                        "imageUrl": relative_image_url,
                        "n": n
                    }
                }
                logger.info(f"[{request_id}] Clipdrop variation payload: {json.dumps(payload, indent=2)}")
                variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=MIDJOURNEY_TIMEOUT)
                if variation_response.status_code != 200:
                    logger.error(f"[{request_id}] Clipdrop variation failed: {variation_response.status_code}, {variation_response.text}")
                    continue
                variation_data = variation_response.json()
                if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                    result_object = variation_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
                    if isinstance(result_object, list):
                        variation_urls.extend(result_object)
                    elif isinstance(result_object, str):
                        variation_urls.append(result_object)
                elif "resultObject" in variation_data:
                    result_object = variation_data["resultObject"]
                    if isinstance(result_object, list):
                        variation_urls.extend(result_object)
                    elif isinstance(result_object, str):
                        variation_urls.append(result_object)
                if variation_urls:
                    logger.info(f"[{request_id}] Successfully created {len(variation_urls)} variations with Clipdrop")
                    break
                else:
                    logger.warning(f"[{request_id}] No variation URLs found in Clipdrop response")
            # Если предыдущий блок не сработал, пытаемся ещё раз через основной URL
            variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload,
                                             timeout=(MIDJOURNEY_TIMEOUT if model.startswith("midjourney") else DEFAULT_TIMEOUT))
            if variation_response.status_code != 200:
                logger.error(f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
                # When the Gateway Timeout (504) error, we return the error immediately, and do not continue to process
                if variation_response.status_code == 504:
                    logger.error(f"[{request_id}] Midjourney API timeout (504). Returning error to client instead of fallback.")
                    return jsonify({
                        "error": "Gateway Timeout (504) occurred while processing image variation request. Try again later."
                    }), 504
                continue
            variation_data = variation_response.json()
            if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                result_object = variation_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
                if isinstance(result_object, list):
                    variation_urls.extend(result_object)
                elif isinstance(result_object, str):
                    variation_urls.append(result_object)
            elif "resultObject" in variation_data:
                result_object = variation_data["resultObject"]
                if isinstance(result_object, list):
                    variation_urls.extend(result_object)
                elif isinstance(result_object, str):
                    variation_urls.append(result_object)
            if variation_urls:
                logger.info(f"[{request_id}] Successfully created {len(variation_urls)} variations with {model}")
                break
            else:
                logger.warning(f"[{request_id}] No variation URLs found in response for model {model}")
        except Exception as e:
            logger.error(f"[{request_id}] Error with model {model}: {str(e)}")
            continue

    try:
        os.unlink(temp_file.name)
    except Exception:
        pass

    if not variation_urls:
        session.close()
        return jsonify({"error": "Failed to create image variations with any available model"}), 500

    full_variation_urls = []
    asset_host = "https://asset.1min.ai"
    for url in variation_urls:
        if not url:
            continue
        relative_url = url.split('asset.1min.ai/', 1)[-1] if "asset.1min.ai/" in url else url.lstrip('/')
        full_url = get_full_url(url)
        full_variation_urls.append({"relative_path": relative_url, "full_url": full_url})

    openai_data = [{"url": data["relative_path"]} for data in full_variation_urls]
    if len(full_variation_urls) == 1:
        markdown_text = f"![Variation]({full_variation_urls[0]['full_url']}) `[_V1_]`"
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** and send it (paste) in the next **prompt**"
    else:
        image_lines = [f"![Variation {i+1}]({data['full_url']}) `[_V{i+1}_]`" for i, data in enumerate(full_variation_urls)]
        markdown_text = "\n".join(image_lines)
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**"

    openai_response = {
        "created": int(time.time()),
        "data": openai_data,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": markdown_text
            },
            "index": 0,
            "finish_reason": "stop"
        }]
    }
    session.close()
    logger.info(f"[{request_id}] Successfully generated {len(openai_data)} image variations using model {current_model}")
    return jsonify(openai_response), 200



