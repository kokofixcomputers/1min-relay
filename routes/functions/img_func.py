# version 1.0.1 #increment every time you make changes
# routes/functions/img_func.py 

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
from routes.functions.shared_func import extract_image_urls, get_full_url
from flask import jsonify, request
import math
import base64
import traceback
from utils.model_mapping import canonicalize_image_model_name, choose_variator_model

from .file_func import upload_asset, get_mime_type
from .shared_func import format_image_response

#===========================================================#
# ----------- Функции для работы с изображениями -----------#
#===========================================================#

def build_generation_payload(model, prompt, request_data, negative_prompt, aspect_ratio, size, mode, request_id):
    """Build payload for image generation based on model."""
    payload = {}
    model = canonicalize_image_model_name(model)
    # Normalize model aliases (e.g., Midjourney -> Magic Art).
    alias_target = IMAGE_MODEL_ALIASES.get(model)
    if alias_target:
        logger.info(f"[{request_id}] Image model alias: {model} -> {alias_target}")
        model = alias_target
    if model == "dall-e-3":
        # Проверяем, входит ли размер в список разрешенных для DALL-E 3
        gen_size = size or request_data.get("size", "1024x1024")
        if gen_size not in DALLE3_SIZES:
            logger.warning(f"[{request_id}] Размер {gen_size} не входит в список разрешенных для DALL-E 3. Используем {DALLE3_SIZES[0]}")
            gen_size = DALLE3_SIZES[0]
            
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dall-e-3",
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                "size": gen_size,
                "quality": request_data.get("quality", "standard"),
                "style": request_data.get("style", "vivid"),
            },
        }
    elif model == "dall-e-2":
        # Проверяем, входит ли размер в список разрешенных для DALL-E 2
        gen_size = size or request_data.get("size", "1024x1024")
        if gen_size not in DALLE2_SIZES:
            logger.warning(f"[{request_id}] Размер {gen_size} не входит в список разрешенных для DALL-E 2. Используем {DALLE2_SIZES[0]}")
            gen_size = DALLE2_SIZES[0]
            
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dall-e-2",
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                "size": gen_size,
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
    elif model in ["stable-image", "stable-image-ultra"]:
        # Stable Image Core / Ultra (1min.ai docs use stable_image_core_* keys for Core)
        # See: https://docs.1min.ai/docs/api/ai-for-image/image-generator/stable-diffusion-core-image-generation
        prompt_obj = {"prompt": prompt}
        if negative_prompt or request_data.get("negativePrompt"):
            prompt_obj["negativePrompt"] = negative_prompt or request_data.get("negativePrompt", "")

        ar_val = aspect_ratio or request_data.get("aspect_ratio") or "1:1"
        out_fmt = request_data.get("output_format") or request_data.get("stable_image_output_format") or "png"
        seed_val = request_data.get("seed")
        style_preset = request_data.get("style_preset") or request_data.get("stable_image_style_preset")

        if model == "stable-image":
            prompt_obj["stable_image_core_aspect_ratio"] = ar_val
            prompt_obj["stable_image_core_output_format"] = out_fmt
            if seed_val is not None:
                prompt_obj["stable_image_core_seed"] = seed_val
            if style_preset:
                prompt_obj["stable_image_core_style_preset"] = style_preset
        else:
            # Ultra docs naming may differ; keep a conservative payload and pass only common fields + aspect ratio
            # (unknown keys can be rejected upstream).
            prompt_obj["aspect_ratio"] = ar_val
            prompt_obj["output_format"] = request_data.get("output_format", "png")
            if seed_val is not None:
                prompt_obj["seed"] = seed_val
            if style_preset:
                prompt_obj["style_preset"] = style_preset

        payload = {"type": "IMAGE_GENERATOR", "model": model, "promptObject": prompt_obj}
    elif model in ["gpt-image-1", "gpt-image-1-mini"]:
        # OpenAI image models via 1min.ai Image Generator
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model,
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                # OpenAI-like images API uses size, keep it when provided.
                "size": size or request_data.get("size", "1024x1024"),
            },
        }
    elif model in ["magic-art", "magic-art_6_1", "magic-art_7_0"]:
        # Magic Art models use aspect_width/aspect_height and n fixed at 4.
        # Docs example: https://docs.1min.ai/docs/api/ai-for-image/image-generator/magic-art-5.2-image-generation
        try:
            ar_parts = tuple(map(int, (aspect_ratio or "1:1").split(":")))
        except Exception:
            ar_parts = (1, 1)
        prompt_obj = {
            "prompt": prompt,
            "mode": mode or request_data.get("mode", "fast"),
            "n": 4,
            "isNiji6": request_data.get("isNiji6", False),
            "maintainModeration": request_data.get("maintainModeration", True),
            "aspect_width": ar_parts[0],
            "aspect_height": ar_parts[1],
        }
        # Optional Magic Art params (pass-through if provided)
        if negative_prompt or request_data.get("negativePrompt"):
            prompt_obj["negativePrompt"] = negative_prompt or request_data.get("negativePrompt", "")
        for k in [
            "no",
            "image_weight",
            "seed",
            "tile",
            "stylize",
            "chaos",
            "weird",
            "character_reference",
            "style_reference",
            # Magic Art 7.0 omni-reference fields
            "omni_reference",
            "omni_reference_weight",
        ]:
            if k in request_data and request_data.get(k) is not None and request_data.get(k) != "":
                prompt_obj[k] = request_data.get(k)
        payload = {"type": "IMAGE_GENERATOR", "model": model, "promptObject": prompt_obj}
    elif model == "dzine":
        # Dzine requires style_code/style_base_model/quality/output_format (docs).
        # https://docs.1min.ai/docs/api/ai-for-image/image-generator/dzine-image-generation
        style_code = request_data.get("style_code")
        style_base_model = request_data.get("style_base_model")
        quality = request_data.get("quality")
        output_format = request_data.get("output_format")
        n_val = request_data.get("n", 1)
        try:
            n_val = int(n_val)
        except Exception:
            n_val = 1
        n_val = max(1, min(4, n_val))

        # If required fields are missing, try to pick "No Style" automatically using Dzine styles API.
        if not style_code or not style_base_model:
            try:
                # api_key is expected to be provided via request headers (OpenAI-like Authorization)
                auth_header = request.headers.get("Authorization", "")
                api_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
                if api_key:
                    cache_key = "dzine:default_style"
                    cached = safe_memcached_operation('get', cache_key)
                    default_style = None
                    if cached:
                        try:
                            if isinstance(cached, (bytes, bytearray)):
                                cached = cached.decode("utf-8")
                            default_style = json.loads(cached) if isinstance(cached, str) else cached
                        except Exception:
                            default_style = None
                    if not default_style:
                        sess = create_session()
                        try:
                            resp = api_request("GET", ONE_MIN_DZINE_STYLES_URL, headers={"API-KEY": api_key}, timeout=DEFAULT_TIMEOUT)
                            if resp.status_code == 200:
                                data = resp.json()
                                styles = data.get("styleList", []) if isinstance(data, dict) else []
                                pick = None
                                for s in styles:
                                    if isinstance(s, dict) and "name" in s and "no style" in str(s["name"]).lower():
                                        pick = s
                                        break
                                if not pick and styles:
                                    pick = styles[0] if isinstance(styles[0], dict) else None
                                if pick:
                                    default_style = {
                                        "style_code": pick.get("style_code"),
                                        "style_base_model": pick.get("base_model"),
                                    }
                                    safe_memcached_operation('set', cache_key, json.dumps(default_style), expiry=3600 * 24)
                        finally:
                            try:
                                sess.close()
                            except Exception:
                                pass
                    if default_style:
                        style_code = style_code or default_style.get("style_code")
                        style_base_model = style_base_model or default_style.get("style_base_model")
            except Exception:
                pass

        if not style_code or not style_base_model:
            return None, (jsonify({
                "error": "Dzine requires 'style_code' and 'style_base_model'. Provide them (or ensure API key can access /api/dzine/styles)."
            }), 400)

        payload = {
            "type": "IMAGE_GENERATOR",
            "model": "dzine",
            "promptObject": {
                "prompt": prompt,
                "style_code": style_code,
                "style_base_model": style_base_model,
                "quality": quality or "STANDARD",
                "n": n_val,
                "output_format": output_format or "webp",
                "size": size or request_data.get("size", "1024x1024"),
            },
        }
        if request_data.get("style_intensity") is not None:
            payload["promptObject"]["style_intensity"] = request_data.get("style_intensity")
        if request_data.get("seed") is not None:
            payload["promptObject"]["seed"] = request_data.get("seed")
        if request_data.get("face_match") is not None:
            payload["promptObject"]["face_match"] = request_data.get("face_match")
        if request_data.get("face_match_image"):
            payload["promptObject"]["face_match_image"] = request_data.get("face_match_image")
    elif model == "recraft":
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model,
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
            },
        }
        if negative_prompt or request_data.get("negativePrompt"):
            payload["promptObject"]["negativePrompt"] = negative_prompt or request_data.get("negativePrompt", "")
    elif model in ["gemini-2.5-flash-image", "gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"]:
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model,
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
            },
        }
    elif model in ["grok-2-image-1212"]:
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model,
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
            },
        }
    elif model in ["qwen-image", "qwen-image-plus", "qwen-image-max"]:
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model,
            "promptObject": {
                "prompt": prompt,
                "n": request_data.get("n", 1),
                "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
            },
        }
    # NOTE: Midjourney models are handled via IMAGE_MODEL_ALIASES -> Magic Art above.
    elif model in ["black-forest-labs/flux-schnell", "flux-schnell",
                   "black-forest-labs/flux-dev", "flux-dev",
                   "black-forest-labs/flux-pro", "flux-pro",
                   "black-forest-labs/flux-1.1-pro", "flux-1.1-pro",
                   "black-forest-labs/flux-1.1-pro-ultra", "flux-1.1-pro-ultra",
                   "black-forest-labs/flux-krea-dev", "flux-krea-dev",
                   "black-forest-labs/flux-schnell-lora", "flux-schnell-lora",
                   "black-forest-labs/flux-dev-lora", "flux-dev-lora"]:
        # Всегда добавляем префикс black-forest-labs/ для моделей flux, если его нет
        model_name = model
        if not model.startswith("black-forest-labs/"):
            model_name = f"black-forest-labs/{model}"
            logger.debug(f"[{request_id}] Добавлен префикс к модели Flux: {model_name}")
            
        payload = {
            "type": "IMAGE_GENERATOR",
            "model": model_name,
            "promptObject": {
                "prompt": prompt,
                "num_outputs": request_data.get("n", 1),
                "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                "output_format": request_data.get("output_format", "webp"),
            },
        }
        if negative_prompt or request_data.get("negativePrompt"):
            payload["promptObject"]["negativePrompt"] = negative_prompt or request_data.get("negativePrompt", "")
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
            
            # Проверяем, входит ли соотношение сторон в разрешенные для модели
            if model in ["midjourney", "midjourney_6_1"] and aspect_ratio not in MIDJOURNEY_ALLOWED_ASPECT_RATIOS:
                ar_error = f"Аспектное соотношение {aspect_ratio} не поддерживается для модели {model}. Разрешенные значения: {', '.join(MIDJOURNEY_ALLOWED_ASPECT_RATIOS)}"
                logger.error(f"[{request_id}] {ar_error}")
                return prompt, None, size, ar_error, mode
                
            # Проверяем для моделей Flux
            if (model.startswith("flux") or model.startswith("black-forest-labs/flux")) and aspect_ratio not in FLUX_ALLOWED_ASPECT_RATIOS:
                ar_error = f"Аспектное соотношение {aspect_ratio} не поддерживается для модели {model}. Разрешенные значения: {', '.join(FLUX_ALLOWED_ASPECT_RATIOS)}"
                logger.error(f"[{request_id}] {ar_error}")
                return prompt, None, size, ar_error, mode
                
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
            
            # Проверяем, входит ли соотношение сторон в разрешенные для модели
            if model in ["midjourney", "midjourney_6_1"] and aspect_ratio not in MIDJOURNEY_ALLOWED_ASPECT_RATIOS:
                ar_error = f"Аспектное соотношение {aspect_ratio} не поддерживается для модели {model}. Разрешенные значения: {', '.join(MIDJOURNEY_ALLOWED_ASPECT_RATIOS)}"
                logger.error(f"[{request_id}] {ar_error}")
                return prompt, None, size, ar_error, mode
                
            # Проверяем для моделей Flux
            if (model.startswith("flux") or model.startswith("black-forest-labs/flux")) and aspect_ratio not in FLUX_ALLOWED_ASPECT_RATIOS:
                ar_error = f"Аспектное соотношение {aspect_ratio} не поддерживается для модели {model}. Разрешенные значения: {', '.join(FLUX_ALLOWED_ASPECT_RATIOS)}"
                logger.error(f"[{request_id}] {ar_error}")
                return prompt, None, size, ar_error, mode
            
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
    
    user_model = canonicalize_image_model_name(user_model)
    # Normalize model aliases (e.g., Midjourney -> Magic Art) BEFORE variation checks.
    alias_target = IMAGE_MODEL_ALIASES.get(user_model)
    if alias_target:
        logger.info(f"[{request_id}] Image variation model alias: {user_model} -> {alias_target}")
        user_model = alias_target

    # Variations are allowed only for models that are BOTH generator+variator.
    # If a generator name doesn't match a variator name, map/fallback to a safe variator model.
    chosen_model = choose_variator_model(user_model, supported_variators=set(IMAGE_VARIATION_MODELS))
    if not chosen_model:
        logger.warning(f"[{request_id}] Model {user_model} does not support variations (no supported variator)")
        return jsonify({"error": f"Model '{user_model}' does not support image variations"}), 400
    if chosen_model != user_model:
        logger.info(f"[{request_id}] Variator model fallback/mapping: {user_model} -> {chosen_model}")
    user_model = chosen_model

    # Try ONLY the requested model (no fallbacks/substitutions).
    variation_models = [user_model]
    logger.info(f"[{request_id}] Trying image variations with model: {variation_models}")
    
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
                    elif model in ["dzine", "recraft", "magic-art", "magic-art_6_1", "magic-art_7_0", "flux-redux-dev", "flux-redux-schnell"]:
                        # Common 1min.ai Image Variator models (per docs list). Keep payload minimal.
                        payload = {
                            "type": "IMAGE_VARIATOR",
                            "model": model,
                            "promptObject": {
                                "imageUrl": image_path,
                                "n": n,
                            },
                        }
                        logger.info(f"[{request_id}] Variation payload for {model}: {json.dumps(payload, indent=2)}")
                    
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
