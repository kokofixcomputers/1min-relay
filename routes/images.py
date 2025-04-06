# routes/images.py

from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request
from utils.memcached import safe_memcached_operation
from . import app, limiter, IMAGE_CACHE, MAX_CACHE_SIZE
from .utils import (
    validate_auth, 
    handle_api_error, 
    upload_asset,
    get_mime_type,
    extract_image_urls,
    format_image_response,
    prepare_image_payload
)


# ----------------------- Utility Functions -----------------------

def get_full_url(url, asset_host="https://asset.1min.ai"):
    """Return full URL based on asset host."""
    if not url.startswith("http"):
        return f"{asset_host}{url}" if url.startswith("/") else f"{asset_host}/{url}"
    return url


def build_generation_payload(model, prompt, request_data, negative_prompt, aspect_ratio, size, mode, request_id):
    """Build payload for image generation based on model."""
    payload = {}
    if model == "dall-e-3":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dall-e-3",
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                "size": size or request_data.get("size", "1024x1024"),
                "quality": request_data.get("quality", "standard"),
                "style": request_data.get("style", "vivid"),
            },
        }
    elif model == "dall-e-2":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dall-e-2",
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                "size": size or request_data.get("size", "1024x1024"),
            },
        }
    elif model == "stable-diffusion-xl-1024-v1-0":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "stable-diffusion-xl-1024-v1-0",
            "promptObject": {
                "prompt": prompt,
                "samples": request_data.get("n", 1),
                "size": size or request_data.get("size", "1024x1024"),
                "cfg_scale": request_data.get("cfg_scale", 7),
                "clip_guidance_preset": request_data.get("clip_guidance_preset", "NONE"),
                "seed": request_data.get("seed", 0),
                "steps": request_data.get("steps", 30),
            },
        }
    elif model in ["midjourney", "midjourney_6_1"]:
        # Parse aspect ratio parts (default 1:1)
        try:
            ar_parts = tuple(map(int, aspect_ratio.split(":"))) if aspect_ratio else (1, 1)
        except Exception:
            ar_parts = (1, 1)
        model_name = "midjourney" if model == "midjourney" else "midjourney_6_1"
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model_name,
            "promptObject": {
                "prompt": prompt,
                "mode": mode or request_data.get("mode", "fast"),
                "n": 4,
                "aspect_width": ar_parts[0],
                "aspect_height": ar_parts[1],
                "isNiji6": request_data.get("isNiji6", False),
                "maintainModeration": request_data.get("maintainModeration", True),
                "image_weight": request_data.get("image_weight", 1),
                "weird": request_data.get("weird", 0),
            },
        }
        if negative_prompt or request_data.get("negativePrompt"):
            payload["promptObject"]["negativePrompt"] = negative_prompt or request_data.get("negativePrompt", "")
        if request_data.get("no", ""):
            payload["promptObject"]["no"] = request_data.get("no", "")
    elif model in ["black-forest-labs/flux-schnell", "flux-schnell",
                   "black-forest-labs/flux-dev", "flux-dev",
                   "black-forest-labs/flux-pro", "flux-pro",
                   "black-forest-labs/flux-1.1-pro", "flux-1.1-pro"]:
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model.split("/")[-1] if "/" in model else model,
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                "output_format": request_data.get("output_format", "webp"),
            },
        }
    elif model in [
        "6b645e3a-d64f-4341-a6d8-7a3690fbf042", "phoenix",
        "b24e16ff-06e3-43eb-8d33-4416c2d75876", "lightning-xl",
        "5c232a9e-9061-4777-980a-ddc8e65647c6", "vision-xl",
        "e71a1c2f-4f80-4800-934f-2c68979d8cc8", "anime-xl",
        "1e60896f-3c26-4296-8ecc-53e2afecc132", "diffusion-xl",
        "aa77f04e-3eec-4034-9c07-d0f619684628", "kino-xl",
        "2067ae52-33fd-4a82-bb92-c2c55e7d2786", "albedo-base-xl"
    ]:
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model,
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 4),
                "size": size,
                "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
            },
        }
        # Удаляем пустые параметры
        if not payload["promptObject"]["negativePrompt"]:
            del payload["promptObject"]["negativePrompt"]
        if model == "e71a1c2f-4f80-4800-934f-2c68979d8cc8":
            payload["promptObject"]["size"] = size or request_data.get("size", "1024x1024")
            payload["promptObject"]["aspect_ratio"] = aspect_ratio
            if not payload["promptObject"]["aspect_ratio"]:
                del payload["promptObject"]["aspect_ratio"]
    else:
        logger.error(f"[{request_id}] Invalid model: {model}")
        return None, ERROR_HANDLER(1002, model)
    return payload, None


def extract_image_urls_from_response(response_json, request_id):
    """Extract image URLs from API response."""
    image_urls = []
    result_object = response_json.get("aiRecord", {}).get("aiRecordDetail", {}).get("resultObject", [])
    if isinstance(result_object, list) and result_object:
        image_urls = result_object
    elif result_object and isinstance(result_object, str):
        image_urls = [result_object]
    if not image_urls and "resultObject" in response_json:
        result = response_json["resultObject"]
        if isinstance(result, list):
            image_urls = result
        elif isinstance(result, str):
            image_urls = [result]
    if not image_urls:
        logger.error(f"[{request_id}] Could not extract image URLs from API response: {json.dumps(response_json)[:500]}")
    return image_urls


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
            return jsonify({"error": "A prompt is required to generate an image"}), 400

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

        image_urls = extract_image_urls_from_response(api_response, request_id)
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
        logger.error(f"[{request_id}] Exception during image generation request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/images/variations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
@cross_origin()
def image_variations():
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
                variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=300)
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
                variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=300)
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


# ----------------------- Helper Functions -----------------------

def parse_aspect_ratio(prompt, model, request_data, request_id=None):
    """
    Extract aspect ratio and mode from prompt.
    Returns: (modified prompt, aspect_ratio, size, error_message, mode)
    """
    aspect_ratio = None
    size = request_data.get("size", "1024x1024")
    ar_error = None
    mode = None

    # Extract mode parameter (--fast или --relax)
    mode_match = re.search(r'(--|\u2014)(fast|relax)\s*', prompt)
    if mode_match:
        mode = mode_match.group(2)
        prompt = re.sub(r'(--|\u2014)(fast|relax)\s*', '', prompt).strip()
        logger.debug(f"[{request_id}] Extracted mode from prompt: {mode}")

    # Extract aspect ratio (--ar width:height)
    ar_match = re.search(r'(--|\u2014)ar\s+(\d+):(\d+)', prompt)
    if ar_match:
        width = int(ar_match.group(2))
        height = int(ar_match.group(3))
        if max(width, height) / min(width, height) > 2:
            ar_error = "Aspect ratio cannot exceed 2:1 or 1:2"
            logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error, mode
        if width < 1 or width > 10000 or height < 1 or height > 10000:
            ar_error = "Aspect ratio values must be between 1 and 10000"
            logger.error(f"[{request_id}] Invalid aspect ratio values: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error, mode
        aspect_ratio = f"{width}:{height}"
        prompt = re.sub(r'(--|\u2014)ar\s+\d+:\d+\s*', '', prompt).strip()
        logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_ratio}")
    elif "aspect_ratio" in request_data:
        aspect_ratio = request_data.get("aspect_ratio")
        if not re.match(r'^\d+:\d+$', aspect_ratio):
            ar_error = "Aspect ratio must be in format width:height"
            logger.error(f"[{request_id}] Invalid aspect ratio format: {aspect_ratio} - {ar_error}")
            return prompt, None, size, ar_error, mode
        width, height = map(int, aspect_ratio.split(':'))
        if max(width, height) / min(width, height) > 2:
            ar_error = "Aspect ratio cannot exceed 2:1 or 1:2"
            logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error, mode
        if width < 1 or width > 10000 or height < 1 or height > 10000:
            ar_error = "Aspect ratio values must be between 1 and 10000"
            logger.error(f"[{request_id}] Invalid aspect ratio values: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error, mode
        logger.debug(f"[{request_id}] Using aspect ratio from request: {aspect_ratio}")

    # Remove negative prompt parameters
    prompt = re.sub(r'(--|\u2014)no\s+.*?(?=(--|\u2014)|$)', '', prompt).strip()

    # Special adjustments for DALL-E 3
    if model == "dall-e-3" and aspect_ratio:
        width, height = map(int, aspect_ratio.split(':'))
        if abs(width / height - 1) < 0.1:
            size = "1024x1024"
            aspect_ratio = "square"
        elif width > height:
            size = "1792x1024"
            aspect_ratio = "landscape"
        else:
            size = "1024x1792"
            aspect_ratio = "portrait"
        logger.debug(f"[{request_id}] Adjusted size for DALL-E 3: {size}, aspect_ratio: {aspect_ratio}")
    elif model in [
        "6b645e3a-d64f-4341-a6d8-7a3690fbf042", "phoenix",
        "b24e16ff-06e3-43eb-8d33-4416c2d75876", "lightning-xl",
        "5c232a9e-9061-4777-980a-ddc8e65647c6", "vision-xl",
        "e71a1c2f-4f80-4800-934f-2c68979d8cc8", "anime-xl",
        "1e60896f-3c26-4296-8ecc-53e2afecc132", "diffusion-xl",
        "aa77f04e-3eec-4034-9c07-d0f619684628", "kino-xl",
        "2067ae52-33fd-4a82-bb92-c2c55e7d2786", "albedo-base-xl"
    ] and aspect_ratio:
        if aspect_ratio == "1:1":
            size = LEONARDO_SIZES["1:1"]
        elif aspect_ratio == "4:3":
            size = LEONARDO_SIZES["4:3"]
        elif aspect_ratio == "3:4":
            size = LEONARDO_SIZES["3:4"]
        else:
            width, height = map(int, aspect_ratio.split(':'))
            ratio = width / height
            if abs(ratio - 1) < 0.1:
                size = LEONARDO_SIZES["1:1"]
                aspect_ratio = "1:1"
            elif ratio > 1:
                size = LEONARDO_SIZES["4:3"]
                aspect_ratio = "4:3"
            else:
                size = LEONARDO_SIZES["3:4"]
                aspect_ratio = "3:4"
        logger.debug(f"[{request_id}] Adjusted size for Leonardo model: {size}, aspect_ratio: {aspect_ratio}")

    return prompt, aspect_ratio, size, ar_error, mode


def retry_image_upload(image_url, api_key, request_id=None):
    """Uploads an image with repeated attempts, returns a direct link to it."""
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Uploading image: {image_url}")
    session = create_session()
    temp_file_path = None
    try:
        if image_url.startswith(("http://", "https://")):
            logger.debug(f"[{request_id}] Fetching image from URL: {image_url}")
            response = session.get(image_url, stream=True)
            response.raise_for_status()
            image_data = response.content
        else:
            logger.debug(f"[{request_id}] Decoding base64 image")
            image_data = base64.b64decode(image_url.split(",")[1])
        if len(image_data) == 0:
            logger.error(f"[{request_id}] Empty image data")
            return None
        temp_file_path = safe_temp_file("image", request_id)
        with open(temp_file_path, "wb") as f:
            f.write(image_data)
        if os.path.getsize(temp_file_path) == 0:
            logger.error(f"[{request_id}] Empty image file created: {temp_file_path}")
            return None
        try:
            with open(temp_file_path, "rb") as f:
                upload_response = session.post(
                    ONE_MIN_ASSET_URL,
                    headers={"API-KEY": api_key},
                    files={"asset": (os.path.basename(image_url),
                                     f,
                                     "image/webp" if image_url.endswith(".webp") else "image/jpeg")}
                )
                if upload_response.status_code != 200:
                    logger.error(f"[{request_id}] Upload failed with status {upload_response.status_code}: {upload_response.text}")
                    return None
                upload_data = upload_response.json()
                if isinstance(upload_data, str):
                    try:
                        upload_data = json.loads(upload_data)
                    except:
                        logger.error(f"[{request_id}] Failed to parse upload response: {upload_data}")
                        return None
                logger.debug(f"[{request_id}] Upload response: {upload_data}")
                if "fileContent" in upload_data and "path" in upload_data["fileContent"]:
                    url = upload_data["fileContent"]["path"]
                    logger.info(f"[{request_id}] Image uploaded successfully: {url}")
                    return url
                logger.error(f"[{request_id}] No path found in upload response")
                return None
        except Exception as e:
            logger.error(f"[{request_id}] Exception during image upload: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"[{request_id}] Exception during image processing: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        session.close()
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"[{request_id}] Removed temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to remove temp file {temp_file_path}: {str(e)}")


def create_image_variations(image_url, user_model, n, aspect_width=None, aspect_height=None, mode=None, request_id=None):
    """
    Creates variations based on the original image, taking into account the specifics of each model.
    """
    variation_urls = []
    current_model = None
    if request_id is None:
        request_id = str(uuid.uuid4())
    generation_params = None
    if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
        try:
            gen_key = f"gen_params:{request_id}"
            params_json = safe_memcached_operation('get', gen_key)
            if params_json:
                if isinstance(params_json, str):
                    generation_params = json.loads(params_json)
                elif isinstance(params_json, bytes):
                    generation_params = json.loads(params_json.decode('utf-8'))
                logger.debug(f"[{request_id}] Retrieved generation parameters from memcached: {generation_params}")
        except Exception as e:
            logger.error(f"[{request_id}] Error retrieving generation parameters: {str(e)}")
    if generation_params:
        if "aspect_width" in generation_params and "aspect_height" in generation_params:
            aspect_width = generation_params.get("aspect_width")
            aspect_height = generation_params.get("aspect_height")
            logger.debug(f"[{request_id}] Using saved aspect ratio: {aspect_width}:{aspect_height}")
        if "mode" in generation_params:
            mode = generation_params.get("mode")
            logger.debug(f"[{request_id}] Using saved mode: {mode}")
    variation_models = []
    if user_model in VARIATION_SUPPORTED_MODELS:
        variation_models.append(user_model)
    variation_models.extend([m for m in ["midjourney_6_1", "midjourney", "clipdrop", "dall-e-2"] if m != user_model])
    variation_models = list(dict.fromkeys(variation_models))
    logger.info(f"[{request_id}] Trying image variations with models: {variation_models}")
    session = create_session()
    try:
        image_response = session.get(image_url, stream=True, timeout=60)
        if image_response.status_code != 200:
            logger.error(f"[{request_id}] Failed to download image: {image_response.status_code}")
            return jsonify({"error": "Failed to download image"}), 500
        for model in variation_models:
            current_model = model
            logger.info(f"[{request_id}] Trying model: {model} for image variations")
            try:
                # Determine MIME type and extension
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

                # Upload image to server
                files = {"asset": (f"variation.{ext}", image_response.content, content_type)}
                upload_response = session.post(ONE_MIN_ASSET_URL, files=files, headers=headers)
                if upload_response.status_code != 200:
                    logger.error(f"[{request_id}] Image upload failed: {upload_response.status_code}")
                    continue
                upload_data = upload_response.json()
                logger.debug(f"[{request_id}] Asset upload response: {upload_data}")
                image_path = None
                if "fileContent" in upload_data and "path" in upload_data["fileContent"]:
                    image_path = upload_data["fileContent"]["path"]
                    if image_path.startswith('/'):
                        image_path = image_path[1:]
                    logger.debug(f"[{request_id}] Using relative path for variation: {image_path}")
                else:
                    logger.error(f"[{request_id}] Could not extract image path from upload response")
                    continue

                if model in ["midjourney_6_1", "midjourney"]:
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": model,
                        "promptObject": {
                            "imageUrl": image_path,
                            "mode": mode or "fast",
                            "n": 4,
                            "isNiji6": False,
                            "aspect_width": aspect_width or 1,
                            "aspect_height": aspect_height or 1,
                            "maintainModeration": True
                        }
                    }
                    logger.info(f"[{request_id}] Midjourney variation payload: {json.dumps(payload['promptObject'], indent=2)}")
                elif model == "dall-e-2":
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "dall-e-2",
                        "promptObject": {
                            "imageUrl": image_path,
                            "n": 1,
                            "size": "1024x1024"
                        }
                    }
                    logger.info(f"[{request_id}] DALL-E 2 variation payload: {json.dumps(payload, indent=2)}")
                    variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=300)
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
                            "imageUrl": image_path,
                            "n": n
                        }
                    }
                    logger.info(f"[{request_id}] Clipdrop variation payload: {json.dumps(payload, indent=2)}")
                    variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=300)
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
                variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload,
                                                 timeout=(MIDJOURNEY_TIMEOUT if model.startswith("midjourney") else DEFAULT_TIMEOUT))
                if variation_response.status_code != 200:
                    logger.error(f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
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

        if not variation_urls:
            logger.error(f"[{request_id}] Failed to create variations with any available model")
            return jsonify({"error": "Failed to create image variations with any available model"}), 500

        openai_response = {"created": int(time.time()), "data": []}
        for url in variation_urls:
            openai_response["data"].append({"url": url})
        text_lines = [f"Image {i} ({url}) [_V{i}_]" for i, url in enumerate(variation_urls, 1)]
        text_lines.append("\n> To generate **variants** of **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**")
        text_response = "\n".join(text_lines)
        openai_response["choices"] = [{
            "message": {"role": "assistant", "content": text_response},
            "index": 0,
            "finish_reason": "stop"
        }]
        logger.info(f"[{request_id}] Returning {len(variation_urls)} variation URLs to client")
        response = jsonify(openai_response)
        return set_response_headers(response)
    except Exception as e:
        logger.error(f"[{request_id}] Exception during image variation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()
