# routes/functions/img_func.py 

from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import (
    ERROR_HANDLER, 
    safe_temp_file,
    create_session, 
    api_request
)
from utils.memcached import safe_memcached_operation
from flask import jsonify, request
import math
import base64
import traceback

from .file_func import upload_asset, get_mime_type
from .shared_func import format_image_response

#===========================================================#
# ----------- Функции для работы с изображениями -----------#
#===========================================================#

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

def extract_image_urls(response_data, request_id=None):
    """
    Извлекает URL изображений из ответа API
    
    Args:
        response_data: Ответ от API
        request_id: ID запроса для логирования
        
    Returns:
        list: Список URL изображений
    """
    image_urls = []
    
    try:
        # Проверяем структуру aiRecord
        if "aiRecord" in response_data and "aiRecordDetail" in response_data["aiRecord"]:
            result = response_data["aiRecord"]["aiRecordDetail"].get("resultObject", [])
            if isinstance(result, list):
                image_urls.extend(result)
            elif isinstance(result, str):
                image_urls.append(result)
                
        # Проверяем прямую структуру resultObject
        elif "resultObject" in response_data:
            result = response_data["resultObject"]
            if isinstance(result, list):
                image_urls.extend(result)
            elif isinstance(result, str):
                image_urls.append(result)
                
        # Проверяем структуру data.url (для Dall-E)
        elif "data" in response_data and isinstance(response_data["data"], list):
            for item in response_data["data"]:
                if "url" in item:
                    image_urls.append(item["url"])
                    
        logger.debug(f"[{request_id}] Extracted {len(image_urls)} image URLs")
        
        if not image_urls:
            logger.error(f"[{request_id}] Could not extract image URLs from API response: {json.dumps(response_data)[:500]}")
            
    except Exception as e:
        logger.error(f"[{request_id}] Error extracting image URLs: {str(e)}")
        
    return image_urls

def parse_aspect_ratio(prompt, model, request_data, request_id=None):
    """
    Parse aspect ratio, size and other parameters from the prompt.
    Enhanced version combining functionality from both implementations.
    Returns: (modified prompt, aspect_ratio, size, error_message, mode)
    """
    original_prompt = prompt
    mode = None
    size = request_data.get("size", "1024x1024")
    aspect_ratio = None
    ar_error = None

    try:
        # Extract mode parameter (--fast or --relax)
        mode_match = re.search(r'(--|\u2014)(fast|relax)\s*', prompt)
        if mode_match:
            mode = mode_match.group(2)
            prompt = re.sub(r'(--|\u2014)(fast|relax)\s*', '', prompt).strip()
            logger.debug(f"[{request_id}] Extracted mode: {mode}")

        # Extract size parameter
        size_match = re.search(r'(--|\u2014)size\s+(\d+x\d+)', prompt)
        if size_match:
            size = size_match.group(2)
            prompt = re.sub(r'(--|\u2014)size\s+\d+x\d+\s*', '', prompt).strip()
            logger.debug(f"[{request_id}] Extracted size: {size}")

        # Extract aspect ratio from prompt
        ar_match = re.search(r'(--|\u2014)ar\s+(\d+):(\d+)', prompt)
        if ar_match:
            width = int(ar_match.group(2))
            height = int(ar_match.group(3))
            
            # Validate aspect ratio
            if width <= 0 or height <= 0:
                logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height}")
                return original_prompt, None, size, "Aspect ratio dimensions must be positive", mode
            
            # Check aspect ratio limits
            if max(width, height) / min(width, height) > 2:
                ar_error = "Aspect ratio cannot exceed 2:1 or 1:2"
                logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height} - {ar_error}")
                return prompt, None, size, ar_error, mode
                
            if width > 10000 or height > 10000:
                ar_error = "Aspect ratio values must be between 1 and 10000"
                logger.error(f"[{request_id}] Invalid aspect ratio values: {width}:{height} - {ar_error}")
                return prompt, None, size, ar_error, mode
            
            # Simplify aspect ratio if needed
            if width > 10 or height > 10:
                gcd_val = math.gcd(width, height)
                width = width // gcd_val
                height = height // gcd_val
            
            aspect_ratio = f"{width}:{height}"
            prompt = re.sub(r'(--|\u2014)ar\s+\d+:\d+\s*', '', prompt).strip()
            logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_ratio}")
        # Check for aspect ratio in request data
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
            
        # Handle special case for dall-e-3 which doesn't support custom aspect ratio
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
        # Special adjustments for Leonardo models
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
    
    except Exception as e:
        logger.error(f"[{request_id}] Error parsing aspect ratio: {str(e)}")
        return original_prompt, None, size, f"Error parsing parameters: {str(e)}", mode

def create_image_variations(image_url, user_model, n, aspect_width=None, aspect_height=None, mode=None, request_id=None):
    """
    Generate variations of the uploaded image using the 1min.ai API.
    Enhanced version combining functionality from both implementations.
    
    Args:
        image_url: URL of the uploaded image
        user_model: Requested model name
        n: Number of variations to generate
        aspect_width: Width for aspect ratio (optional)
        aspect_height: Height for aspect ratio (optional)
        mode: Generation mode (optional)
        request_id: Request ID for logging
        
    Returns:
        list: Image URLs of the generated variations or tuple (response, status_code) in case of error
    """
    # Set request_id if not provided
    if request_id is None:
        request_id = str(uuid.uuid4())[:8]
        
    variation_urls = []
    current_model = None
    
    # Try to get saved generation parameters from memcached
    generation_params = None
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
        
    # Use saved parameters if available
    if generation_params:
        if "aspect_width" in generation_params and "aspect_height" in generation_params:
            aspect_width = generation_params.get("aspect_width")
            aspect_height = generation_params.get("aspect_height")
            logger.debug(f"[{request_id}] Using saved aspect ratio: {aspect_width}:{aspect_height}")
        if "mode" in generation_params:
            mode = generation_params.get("mode")
            logger.debug(f"[{request_id}] Using saved mode: {mode}")
    
    # Determine which models to try for variations
    variation_models = []
    if user_model in VARIATION_SUPPORTED_MODELS:
        variation_models.append(user_model)
    # Add fallback models
    variation_models.extend([m for m in ["midjourney_6_1", "midjourney", "clipdrop", "dall-e-2"] if m != user_model])
    variation_models = list(dict.fromkeys(variation_models))
    logger.info(f"[{request_id}] Trying image variations with models: {variation_models}")
    
    try:
        # Get API key from request
        auth_header = request.headers.get("Authorization", "")
        api_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
        
        if not api_key:
            logger.error(f"[{request_id}] No API key provided for variation")
            return None
            
        headers = {"API-KEY": api_key, "Content-Type": "application/json"}
        session = create_session()
        
        try:
            # Download the image from the URL
            image_response = session.get(image_url, stream=True, timeout=MIDJOURNEY_TIMEOUT)
            if image_response.status_code != 200:
                logger.error(f"[{request_id}] Failed to download image: {image_response.status_code}")
                return jsonify({"error": "Failed to download image"}), 500
                
            # Try each model in sequence
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
                    
                    # Create model-specific payload
                    payload = {}
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
                        
                        # Try DALL-E 2 specific endpoint first
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
                                "imageUrl": image_path,
                                "n": n
                            }
                        }
                        logger.info(f"[{request_id}] Clipdrop variation payload: {json.dumps(payload, indent=2)}")
                        
                        # Try Clipdrop specific endpoint
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
                    
                    # If we reach here for midjourney or if previous attempts didn't succeed, try main API endpoint
                    if payload:
                        timeout = MIDJOURNEY_TIMEOUT if model.startswith("midjourney") else DEFAULT_TIMEOUT
                        
                        # Make the API request
                        variation_response = api_request("POST", ONE_MIN_API_URL, headers=headers, json=payload, timeout=timeout)
                        
                        if variation_response.status_code != 200:
                            logger.error(f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
                            # When the Gateway Timeout (504) error, we return the error immediately, and do not continue to process
                            if variation_response.status_code == 504:
                                logger.error(f"[{request_id}] Midjourney API timeout (504). Returning error to client instead of fallback.")
                                return jsonify({
                                    "error": "Gateway Timeout (504) occurred while processing image variation request. Try again later."
                                }), 504
                            continue
                        
                        # Process the response
                        variation_data = variation_response.json()
                        
                        # Extract variation URLs from response
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
            
            # Handle case where all models failed
            if not variation_urls:
                logger.error(f"[{request_id}] Failed to create variations with any available model")
                return jsonify({"error": "Failed to create image variations with any available model"}), 500
            
            # Format the successful response
            logger.info(f"[{request_id}] Generated {len(variation_urls)} image variations with {current_model}")
            return variation_urls
            
        finally:
            session.close()
            
    except Exception as e:
        logger.error(f"[{request_id}] Error generating image variations: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
