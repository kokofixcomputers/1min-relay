# routes/images.py

# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request, safe_temp_file, calculate_token
from . import app, limiter, MEMORY_STORAGE  # Импортируем app, limiter и MEMORY_STORAGE из модуля routes


# Маршруты для работы с изображениями
@app.route("/v1/images/generations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def generate_image():
    """
    Route for generating images
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    # Create a unique ID for request
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received request: /v1/images/generations")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]
    headers = {"API-KEY": api_key, "Content-Type": "application/json"}

    # Verification that the data is transmitted in the correct format
    if request.is_json:
        request_data = request.get_json()
    else:
        logger.error(f"[{request_id}] Request content-type is not application/json")
        return jsonify({"error": "Content-type must be application/json"}), 400

    # We get the necessary parameters from the request
    model = request_data.get("model", "dall-e-3").strip()
    prompt = request_data.get("prompt", "").strip()

    # If the request was redirected from the Conversation function,
    # We must take only the last request of the user without history
    if request.environ.get("HTTP_REFERER") and "chat/completions" in request.environ.get("HTTP_REFERER"):
        logger.debug(f"[{request_id}] Request came from chat completions, isolating the prompt")
        # We do not combine prompt depths, but we take only the last user request

    # Determine the presence of negative prompts (if any)
    negative_prompt = None
    no_match = re.search(r'(--|\u2014)no\s+(.*?)(?=(--|\u2014)|$)', prompt)
    if no_match:
        negative_prompt = no_match.group(2).strip()
        # We delete negative prompt plate from the main text
        prompt = re.sub(r'(--|\u2014)no\s+.*?(?=(--|\u2014)|$)', '', prompt).strip()

    # We process the ratio of the parties and the size
    prompt, aspect_ratio, size, ar_error, mode = parse_aspect_ratio(prompt, model, request_data, request_id)

    # If there was an error in processing the ratio of the parties, we return it to the user
    if ar_error:
        return jsonify({"error": ar_error}), 400

    # Checking the availability of promptpus
    if not prompt:
        # We check if there is a prompt in messages
        messages = request_data.get("messages", [])
        if messages and len(messages) > 0:
            # We take only the last user message
            last_message = messages[-1]
            if last_message.get("role") == "user":
                content = last_message.get("content", "")
                if isinstance(content, str):
                    prompt = content
                elif isinstance(content, list):
                    # Collect all the text parts of the contents
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                    prompt = " ".join(text_parts)

                # We process the parameters in Prompt from the message
                negative_prompt = None
                no_match = re.search(r'(--|\u2014)no\s+(.*?)(?=(--|\u2014)|$)', prompt)
                if no_match:
                    negative_prompt = no_match.group(2).strip()

                # We re -process the prompt plate to delete modifiers
                prompt, aspect_ratio, size, ar_error, mode = parse_aspect_ratio(prompt, model, request_data, request_id)

                if ar_error:
                    return jsonify({"error": ar_error}), 400

        if prompt:
            logger.debug(f"[{request_id}] Found prompt in messages: {prompt[:100]}..." if len(
                prompt) > 100 else f"[{request_id}] Found prompt in messages: {prompt}")
        else:
            logger.error(f"[{request_id}] No prompt provided")
            return jsonify({"error": "A prompt is required to generate an image"}), 400

    logger.info(f"[{request_id}] Using model: {model}, prompt: '{prompt}'")

    try:
        # Determine the URL for different models
        api_url = f"{ONE_MIN_API_URL}"

        # Tysout 15 minutes for all images generation models
        timeout = MIDJOURNEY_TIMEOUT

        # We form Payload for request depending on the model
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
                    "clip_guidance_preset": request_data.get(
                        "clip_guidance_preset", "NONE"
                    ),
                    "seed": request_data.get("seed", 0),
                    "steps": request_data.get("steps", 30),
                },
            }
        elif model == "stable-diffusion-v1-6":
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "stable-diffusion-v1-6",
                "promptObject": {
                    "prompt": prompt,
                    "samples": request_data.get("n", 1),
                    "cfg_scale": request_data.get("cfg_scale", 7),
                    "clip_guidance_preset": request_data.get(
                        "clip_guidance_preset", "NONE"
                    ),
                    "height": request_data.get("height", 512),
                    "width": request_data.get("width", 512),
                    "seed": request_data.get("seed", 0),
                    "steps": request_data.get("steps", 30),
                },
            }
        elif model in ["midjourney", "midjourney_6_1"]:
            # Permissible parties for the Midjourney

            # Default values
            aspect_width = 1
            aspect_height = 1
            no_param = ""

            # If the ratio of the parties is indicated
            if aspect_ratio:
                # We break the parties to the width and height ratio
                ar_parts = aspect_ratio.split(":")
                aspect_width = int(ar_parts[0])
                aspect_height = int(ar_parts[1])

            model_name = "midjourney" if model == "midjourney" else "midjourney_6_1"

            # Add logistics for the mode
            logger.info(f"[{request_id}] Midjourney generation payload:")
            logger.info(f"[{request_id}] Using mode from prompt: {mode}")

            payload = {
                "type": "IMAGE_GENERATOR",
                "model": model_name,
                "promptObject": {
                    "prompt": prompt,
                    "mode": mode or request_data.get("mode", "fast"),
                    # We use the mode of prompt plate or from REQUEST_DATA
                    "n": 4,  # Midjourney always generates 4 images
                    "aspect_width": aspect_width,
                    "aspect_height": aspect_height,
                    "isNiji6": request_data.get("isNiji6", False),
                    "maintainModeration": request_data.get("maintainModeration", True),
                    "image_weight": request_data.get("image_weight", 1),
                    "weird": request_data.get("weird", 0),
                },
            }

            # Add NegativePrompt and No only if they are not empty
            if negative_prompt or request_data.get("negativePrompt"):
                payload["promptObject"]["negativePrompt"] = negative_prompt or request_data.get("negativePrompt", "")

            no_param = request_data.get("no", "")
            if no_param:
                payload["promptObject"]["no"] = no_param

            # Detailed logging for Midjourney - only once!
            logger.info(f"[{request_id}] Midjourney promptObject: {json.dumps(payload['promptObject'], indent=2)}")
        elif model in ["black-forest-labs/flux-schnell", "flux-schnell"]:
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "black-forest-labs/flux-schnell",
                "promptObject": {
                    "prompt": prompt,
                    "num_outputs": request_data.get("n", 1),
                    "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                    "output_format": request_data.get("output_format", "webp"),
                },
            }
        elif model in ["black-forest-labs/flux-dev", "flux-dev"]:
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "black-forest-labs/flux-dev",
                "promptObject": {
                    "prompt": prompt,
                    "num_outputs": request_data.get("n", 1),
                    "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                    "output_format": request_data.get("output_format", "webp"),
                },
            }
        elif model in ["black-forest-labs/flux-pro", "flux-pro"]:
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "black-forest-labs/flux-pro",
                "promptObject": {
                    "prompt": prompt,
                    "num_outputs": request_data.get("n", 1),
                    "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                    "output_format": request_data.get("output_format", "webp"),
                },
            }
        elif model in ["black-forest-labs/flux-1.1-pro", "flux-1.1-pro"]:
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "black-forest-labs/flux-1.1-pro",
                "promptObject": {
                    "prompt": prompt,
                    "num_outputs": request_data.get("n", 1),
                    "aspect_ratio": aspect_ratio or request_data.get("aspect_ratio", "1:1"),
                    "output_format": request_data.get("output_format", "webp"),
                },
            }
        elif model in [
            "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
            "phoenix",
        ]:  # Leonardo.ai - Phoenix
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 4),
                    "size": size,  # The size is determined on the basis of aspect_ratio in Parse_aspect_ratio
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # We delete empty parameters
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
            logger.debug(
                f"[{request_id}] Leonardo.ai Phoenix payload with size: {size}, from aspect_ratio: {aspect_ratio}")
        elif model in [
            "b24e16ff-06e3-43eb-8d33-4416c2d75876",
            "lightning-xl",
        ]:  # Leonardo.ai - Lightning XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "b24e16ff-06e3-43eb-8d33-4416c2d75876",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 4),
                    "size": size,  # The size is determined on the basis of aspect_ratio in Parse_aspect_ratio
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # We delete empty parameters
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
            logger.debug(
                f"[{request_id}] Leonardo.ai Lightning XL payload with size: {size}, from aspect_ratio: {aspect_ratio}")
        elif model in [
            "5c232a9e-9061-4777-980a-ddc8e65647c6",
            "vision-xl",
        ]:  # Leonardo.ai - Vision XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "5c232a9e-9061-4777-980a-ddc8e65647c6",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 4),
                    "size": size,  # The size is determined on the basis of aspect_ratio in Parse_aspect_ratio
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                },
            }
            # We delete empty parameters
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
            logger.debug(
                f"[{request_id}] Leonardo.ai Vision XL payload with size: {size}, from aspect_ratio: {aspect_ratio}")
        elif model in [
            "e71a1c2f-4f80-4800-934f-2c68979d8cc8",
            "anime-xl",
        ]:  # Leonardo.ai - Anime XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "e71a1c2f-4f80-4800-934f-2c68979d8cc8",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 4),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                    "aspect_ratio": aspect_ratio
                },
            }
            # We delete empty parameters
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
            if not payload["promptObject"]["aspect_ratio"]:
                del payload["promptObject"]["aspect_ratio"]
            logger.debug(f"[{request_id}] Leonardo.ai Anime XL payload with size: {size}")
        elif model in [
            "1e60896f-3c26-4296-8ecc-53e2afecc132",
            "diffusion-xl",
        ]:  # Leonardo.ai - Diffusion XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "1e60896f-3c26-4296-8ecc-53e2afecc132",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 4),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                    "aspect_ratio": aspect_ratio
                },
            }
            # We delete empty parameters
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
            if not payload["promptObject"]["aspect_ratio"]:
                del payload["promptObject"]["aspect_ratio"]
            logger.debug(f"[{request_id}] Leonardo.ai Diffusion XL payload with size: {size}")
        elif model in [
            "aa77f04e-3eec-4034-9c07-d0f619684628",
            "kino-xl",
        ]:  # Leonardo.ai - Kino XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "aa77f04e-3eec-4034-9c07-d0f619684628",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 4),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                    "aspect_ratio": aspect_ratio
                },
            }
            # We delete empty parameters
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
            if not payload["promptObject"]["aspect_ratio"]:
                del payload["promptObject"]["aspect_ratio"]
            logger.debug(f"[{request_id}] Leonardo.ai Kino XL payload with size: {size}")
        elif model in [
            "2067ae52-33fd-4a82-bb92-c2c55e7d2786",
            "albedo-base-xl",
        ]:  # Leonardo.ai - Albedo Base XL
            payload = {
                "type": "IMAGE_GENERATOR",
                "model": "2067ae52-33fd-4a82-bb92-c2c55e7d2786",
                "promptObject": {
                    "prompt": prompt,
                    "n": request_data.get("n", 4),
                    "size": size or request_data.get("size", "1024x1024"),
                    "negativePrompt": negative_prompt or request_data.get("negativePrompt", ""),
                    "aspect_ratio": aspect_ratio
                },
            }
            # We delete empty parameters
            if not payload["promptObject"]["negativePrompt"]:
                del payload["promptObject"]["negativePrompt"]
            if not payload["promptObject"]["aspect_ratio"]:
                del payload["promptObject"]["aspect_ratio"]
            logger.debug(f"[{request_id}] Leonardo.ai Albedo Base XL payload with size: {size}")
        else:
            logger.error(f"[{request_id}] Invalid model: {model}")
            return ERROR_HANDLER(1002, model)

        logger.debug(f"[{request_id}] Sending request to 1min.ai API: {api_url}")
        logger.debug(f"[{request_id}] Payload: {json.dumps(payload)[:500]}")

        # We set parameters for repeated attempts
        max_retries = 1  # Only one attempt for all models
        retry_count = 0
        start_time = time.time()  # We remember the start time to track the total waiting time

        try:
            # We send a request with a timeout
            response = api_request(
                "POST",
                api_url,
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=False
            )

            logger.debug(f"[{request_id}] Response status code: {response.status_code}")

            # If a successful answer is received, we process it
            if response.status_code == 200:
                one_min_response = response.json()
            else:
                # For any errors, we immediately return the answer
                error_msg = "Unknown error"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = error_data["error"]
                except:
                    pass

                if response.status_code == 401:
                    return ERROR_HANDLER(1020, key=api_key)
                else:
                    return (
                        jsonify({"error": error_msg}),
                        response.status_code,
                    )

        except Exception as e:
            logger.error(f"[{request_id}] Exception during API request: {str(e)}")
            return jsonify({"error": f"API request failed: {str(e)}"}), 500

        try:
            # We get all the URL images if they are available
            image_urls = []

            # Check if the response of an array of URL images
            result_object = one_min_response.get("aiRecord", {}).get("aiRecordDetail", {}).get("resultObject", [])

            if isinstance(result_object, list) and result_object:
                image_urls = result_object
            elif result_object and isinstance(result_object, str):
                image_urls = [result_object]

            # If the URL is not found, we will try other extracts
            if not image_urls:
                if "resultObject" in one_min_response:
                    if isinstance(one_min_response["resultObject"], list):
                        image_urls = one_min_response["resultObject"]
                    else:
                        image_urls = [one_min_response["resultObject"]]

            if not image_urls:
                logger.error(
                    f"[{request_id}] Could not extract image URLs from API response: {json.dumps(one_min_response)[:500]}"
                )
                return (
                    jsonify({"error": "Could not extract image URLs from API response"}),
                    500,
                )

            logger.debug(
                f"[{request_id}] Successfully generated {len(image_urls)} images"
            )
            
            # We save the parameters of the image generation in Memcache for subsequent use in variations
            if model in ["midjourney", "midjourney_6_1"]:
                try:
                    # We save the parameters for each generated image
                    for url in image_urls:
                        if url:
                            # We extract ID images from the URL
                            image_id_match = re.search(r'images/(\d+_\d+_\d+_\d+_\d+_\d+|\w+\d+)\.png', url)
                            if image_id_match:
                                image_id = image_id_match.group(1)
                                logger.info(f"[{request_id}] Extracted image_id from URL: {image_id}")
                                
                                # We save only the necessary parameters
                                gen_params = {
                                    "mode": payload["promptObject"].get("mode", "fast"),
                                    "aspect_width": payload["promptObject"].get("aspect_width", 1),
                                    "aspect_height": payload["promptObject"].get("aspect_height", 1),
                                    "isNiji6": payload["promptObject"].get("isNiji6", False),
                                    "maintainModeration": payload["promptObject"].get("maintainModeration", True)
                                }
                                
                                gen_params_key = f"gen_params:{image_id}"
                                # We use the updated version of Safe_Memcache_OPREEN
                                safe_memcached_operation('set', gen_params_key, gen_params, expiry=3600*24*7)  # Store 7 days
                                logger.info(f"[{request_id}] Saved generation parameters for image {image_id}: {gen_params}")
                                
                                # We check that the parameters are precisely preserved correctly
                                if gen_params_key in MEMORY_STORAGE:
                                    logger.info(f"[{request_id}] Verified saved directly in MEMORY_STORAGE: {MEMORY_STORAGE[gen_params_key]}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error saving generation parameters: {str(e)}")

            # We form full URLs for all images
            full_image_urls = []
            asset_host = "https://asset.1min.ai"

            for url in image_urls:
                if not url:
                    continue

                # Check if the URL contains full URL
                if not url.startswith("http"):
                    # If the image begins with /, do not add one more /
                    if url.startswith("/"):
                        full_url = f"{asset_host}{url}"
                    else:
                        full_url = f"{asset_host}/{url}"
                else:
                    full_url = url

                full_image_urls.append(full_url)

            # We form a response in Openai format with teams for variations
            openai_data = []
            for i, url in enumerate(full_image_urls):
                # Create a short identifier for image
                image_id = str(uuid.uuid4())[:8]

                # Add commands for variations only if the model supports variations
                if model in IMAGE_VARIATION_MODELS:
                    variation_commands = {
                        "url": url,
                        "revised_prompt": prompt,
                        "variation_commands": {
                            "variation": f"/v{i + 1} {url}",  # Team to create variation with number
                        }
                    }
                    openai_data.append(variation_commands)
                else:
                    openai_data.append({"url": url, "revised_prompt": prompt})

            openai_response = {
                "created": int(time.time()),
                "data": openai_data,
            }

            # For compatibility with the format of text answers, add Structure_outPut
            structured_output = {"type": "image", "image_urls": full_image_urls}

            # We form a markdown text with variation buttons
            if len(full_image_urls) == 1:
                text_response = f"![Image]({full_image_urls[0]}) `[_V1_]`"
                # Add a hint about the creation of variations
                text_response += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** and send it (paste) in the next **prompt**"
            else:
                # We form a text with images and buttons of variations on one line
                image_lines = []

                for i, url in enumerate(full_image_urls):
                    image_lines.append(f"![Image {i + 1}]({url}) `[_V{i + 1}_]`")

                # Combine lines with a new line between them
                text_response = "\n".join(image_lines)

                # Add a hint about the creation of variations
                text_response += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**"

            openai_response["choices"] = [
                {
                    "message": {
                        "role": "assistant",
                        "content": text_response,
                        "structured_output": structured_output
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]

            logger.info(f"[{request_id}] Returning {len(openai_data)} image URLs to client")
            response = make_response(jsonify(openai_response))
            set_response_headers(response)
            return response, 200
        except Exception as e:
            logger.error(
                f"[{request_id}] Error processing image generation response: {str(e)}"
            )
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(
            f"[{request_id}] Exception during image generation request: {str(e)}"
        )
        return jsonify({"error": str(e)}), 500


@app.route("/v1/images/variations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
@cross_origin()
def image_variations():
    if request.method == "OPTIONS":
        return handle_options_request()

    # Create a unique ID for request
    request_id = str(uuid.uuid4())
    logger.debug(f"[{request_id}] Processing image variation request")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)
    api_key = auth_header.split(" ")[1]

    # We check whether a request has come with the REQUEST_ID parameter (redirection from/V1/Chat/Complets)
    if 'request_id' in request.args:
        # We get data on variation from storages (Memcache or Memory_storage)
        redirect_request_id = request.args.get('request_id')
        variation_key = f"variation:{redirect_request_id}"
        logger.info(f"[{request_id}] Looking for variation data with key: {variation_key}")
        
        variation_data_json = safe_memcached_operation('get', variation_key)
        
        if variation_data_json:
            logger.info(f"[{request_id}] Found variation data: {variation_data_json}")
            try:
                if isinstance(variation_data_json, str):
                    variation_data = json.loads(variation_data_json)
                elif isinstance(variation_data_json, bytes):
                    variation_data = json.loads(variation_data_json.decode('utf-8'))
                else:
                    variation_data = variation_data_json

                # We get the way to the temporary file, model and number of variations
                temp_file_path = variation_data.get("temp_file")
                model = variation_data.get("model")
                n = variation_data.get("n", 1)
                # We get a relative path from the data if it was preserved
                image_path = variation_data.get("image_path")

                logger.debug(
                    f"[{request_id}] Retrieved variation data from memcached: model={model}, n={n}, temp_file={temp_file_path}")
                if image_path:
                    logger.debug(f"[{request_id}] Retrieved image path from memcached: {image_path}")

                # We check that the file exists
                file_exists = os.path.exists(temp_file_path)
                logger.info(f"[{request_id}] Temporary file exists: {file_exists}, path: {temp_file_path}")
                
                if file_exists:
                    # We download the file and process directly
                    try:
                        with open(temp_file_path, 'rb') as f:
                            file_data = f.read()
                        
                        file_size = len(file_data)
                        logger.info(f"[{request_id}] Read temporary file, size: {file_size} bytes")

                        # Create a temporary file for processing a request
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        temp_file.write(file_data)
                        temp_file.close()
                        logger.info(f"[{request_id}] Created new temporary file: {temp_file.name}")

                        # Create a file object for the Image_variations route
                        from io import BytesIO
                        file_data_io = BytesIO(file_data)

                        # We register the file in Request.files via Workraund
                        from werkzeug.datastructures import FileStorage
                        file_storage = FileStorage(
                            stream=file_data_io,
                            filename="variation.png",
                            content_type="image/png",
                        )
                        
                        logger.info(f"[{request_id}] Created FileStorage object for image")

                        # We process a request with a new temporary file
                        request.files = {"image": file_storage}
                        logger.info(f"[{request_id}] Added file to request.files")

                        # Create a form with the necessary parameters
                        form_data = [("model", model), ("n", str(n))]

                        # If there is a relative path, add it to the form
                        if image_path:
                            form_data.append(("image_path", image_path))
                            logger.info(f"[{request_id}] Added image_path to form_data: {image_path}")

                        request.form = MultiDict(form_data)
                        logger.info(f"[{request_id}] Set request.form with data: {form_data}")

                        logger.info(f"[{request_id}] Using file from memcached for image variations")

                        # We delete the original temporary file
                        try:
                            os.unlink(temp_file_path)
                            logger.debug(f"[{request_id}] Deleted original temporary file: {temp_file_path}")
                        except Exception as e:
                            logger.warning(f"[{request_id}] Failed to delete original temporary file: {str(e)}")

                        # We will use the original temporary file instead of creating a new
                        # to avoid problems with the closing of the flow
                    except Exception as e:
                        logger.error(f"[{request_id}] Error processing file from memcached: {str(e)}")
                        return jsonify({"error": f"Error processing variation request: {str(e)}"}), 500
                else:
                    logger.error(f"[{request_id}] Temporary file not found: {temp_file_path}")
                    return jsonify({"error": "Image file not found"}), 400
            except Exception as e:
                logger.error(f"[{request_id}] Error processing variation data: {str(e)}")
                return jsonify({"error": f"Error processing variation request: {str(e)}"}), 500
        else:
            logger.error(f"[{request_id}] No variation data found in memcached with key: {variation_key}")
            return jsonify({"error": "No variation data found"}), 400

    # Getting an image file
    if "image" not in request.files:
        logger.error(f"[{request_id}] No image file provided")
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    original_model = request.form.get("model", "dall-e-2").strip()
    n = int(request.form.get("n", 1))
    size = request.form.get("size", "1024x1024")
    prompt_text = request.form.get("prompt", "")  # We extract the prompt field from the request if it is
    # mode = request.form.get("mode", "relax")  # We get a regime from a request

    # We check whether the relative path to the image in the Form-data has been transmitted
    relative_image_path = request.form.get("image_path")
    if relative_image_path:
        logger.debug(f"[{request_id}] Using relative image path from form: {relative_image_path}")

    logger.debug(f"[{request_id}] Original model requested: {original_model} for image variations")

    # Determine the order of models for Fallback
    fallback_models = ["midjourney_6_1", "midjourney", "clipdrop", "dall-e-2"]

    # If the requested model supports variations, we try it first
    if original_model in IMAGE_VARIATION_MODELS:
        # We start with the requested model, then we try others, excluding the already requested
        models_to_try = [original_model] + [m for m in fallback_models if m != original_model]
    else:
        # If the requested model does not support variations, we start with Fallback models
        logger.warning(
            f"[{request_id}] Model {original_model} does not support image variations. Will try fallback models")
        models_to_try = fallback_models

    # We save a temporary file for multiple use
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image_file.save(temp_file.name)
        temp_file.close()
    except Exception as e:
        logger.error(f"[{request_id}] Failed to save temporary file: {str(e)}")
        return jsonify({"error": "Failed to process image file"}), 500

    # Create a session to download the image
    session = create_session()
    headers = {"API-KEY": api_key}

    # We extract the ratio of the parties from the prompt field if it is
    aspect_width = 1
    aspect_height = 1
    if "--ar" in prompt_text:
        ar_match = re.search(r'--ar\s+(\d+):(\d+)', prompt_text)
        if ar_match:
            aspect_width = int(ar_match.group(1))
            aspect_height = int(ar_match.group(2))
            logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_width}:{aspect_height}")

    # Initialize the variable for variations in front of the cycle
    variation_urls = []
    current_model = None

    # We try each model in turn
    for model in models_to_try:
        logger.info(f"[{request_id}] Trying model: {model} for image variations")
        current_model = model

        try:
            # Special processing for Dall-E 2
            if model == "dall-e-2":
                # For Dall-E 2, you need to use a special Openai and direct file transfer
                logger.debug(f"[{request_id}] Special handling for DALL-E 2 variations")

                # Open the image file and create a request
                with open(temp_file.name, 'rb') as img_file:
                    # Openai expects a file directly to Form-Data
                    dalle_files = {
                        'image': (os.path.basename(temp_file.name), img_file, 'image/png')
                    }

                    # Request parameters
                    dalle_form_data = {
                        'n': n,
                        'size': size,
                        'model': 'dall-e-2'
                    }

                    # We create a request for variation directly to Openai API
                    try:
                        # Try to use a direct connection to Openai if available
                        openai_api_key = os.environ.get("OPENAI_API_KEY")
                        if openai_api_key:
                            openai_headers = {"Authorization": f"Bearer {openai_api_key}"}
                            openai_url = "https://api.openai.com/v1/images/variations"

                            logger.debug(f"[{request_id}] Trying direct OpenAI API for DALL-E 2 variations")
                            variation_response = requests.post(
                                openai_url,
                                files=dalle_files,
                                data=dalle_form_data,
                                headers=openai_headers,
                                timeout=300
                            )

                            if variation_response.status_code == 200:
                                logger.debug(f"[{request_id}] OpenAI API variation successful")
                                variation_data = variation_response.json()

                                # We extract the URL from the answer
                                if "data" in variation_data and isinstance(variation_data["data"], list):
                                    for item in variation_data["data"]:
                                        if "url" in item:
                                            variation_urls.append(item["url"])

                                if variation_urls:
                                    logger.info(
                                        f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2 via OpenAI API")
                                    # We form an answer in Openai API format
                                    response_data = {
                                        "created": int(time.time()),
                                        "data": [{"url": url} for url in variation_urls]
                                    }
                                    return jsonify(response_data)
                            else:
                                logger.error(
                                    f"[{request_id}] OpenAI API variation failed: {variation_response.status_code}, {variation_response.text}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error trying direct OpenAI API: {str(e)}")

                    # If the direct request to Openai failed, we try through 1min.ai API
                    try:
                        # We reject the file because it could be read in the previous request
                        img_file.seek(0)

                        # We draw a request through our own and 1min.ai and dall-e 2
                        onemin_url = "https://api.1min.ai/api/features/images/variations"

                        logger.debug(f"[{request_id}] Trying 1min.ai API for DALL-E 2 variations")
                        dalle_onemin_headers = {"API-KEY": api_key}
                        variation_response = requests.post(
                            onemin_url,
                            files=dalle_files,
                            data=dalle_form_data,
                            headers=dalle_onemin_headers,
                            timeout=300
                        )

                        if variation_response.status_code == 200:
                            logger.debug(f"[{request_id}] 1min.ai API variation successful")
                            variation_data = variation_response.json()

                            # We extract the URL from the answer
                            if "data" in variation_data and isinstance(variation_data["data"], list):
                                for item in variation_data["data"]:
                                    if "url" in item:
                                        variation_urls.append(item["url"])

                            if variation_urls:
                                logger.info(
                                    f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2 via 1min.ai API")
                                # We form an answer in Openai API format
                                response_data = {
                                    "created": int(time.time()),
                                    "data": [{"url": url} for url in variation_urls]
                                }
                                return jsonify(response_data)
                        else:
                            logger.error(
                                f"[{request_id}] 1min.ai API variation failed: {variation_response.status_code}, {variation_response.text}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error trying 1min.ai API: {str(e)}")

                # If you could not create a variation with Dall-E 2, we continue with other models
                logger.warning(f"[{request_id}] Failed to create variations with DALL-E 2, trying next model")
                continue

            # For other models, we use standard logic
            # Image loading in 1min.ai
            with open(temp_file.name, 'rb') as img_file:
                files = {"asset": (os.path.basename(temp_file.name), img_file, "image/png")}

                asset_response = session.post(
                    ONE_MIN_ASSET_URL, files=files, headers=headers
                )
                logger.debug(
                    f"[{request_id}] Image upload response status code: {asset_response.status_code}"
                )

                if asset_response.status_code != 200:
                    logger.error(
                        f"[{request_id}] Failed to upload image: {asset_response.status_code} - {asset_response.text}"
                    )
                    continue  # We try the next model

                # Extract an ID of a loaded image and a full URL
                asset_data = asset_response.json()
                logger.debug(f"[{request_id}] Asset upload response: {asset_data}")

                # We get a URL or ID image
                image_id = None
                image_url = None
                image_location = None

                # We are looking for ID in different places of the response structure
                if "id" in asset_data:
                    image_id = asset_data["id"]
                elif "fileContent" in asset_data and "id" in asset_data["fileContent"]:
                    image_id = asset_data["fileContent"]["id"]
                elif "fileContent" in asset_data and "uuid" in asset_data["fileContent"]:
                    image_id = asset_data["fileContent"]["uuid"]

                # We are looking for an absolute URL (location) for image
                if "asset" in asset_data and "location" in asset_data["asset"]:
                    image_location = asset_data["asset"]["location"]
                    # Extract a relative path if the URL contains the domain
                    if image_location and "asset.1min.ai/" in image_location:
                        image_location = image_location.split('asset.1min.ai/', 1)[1]
                    # Remove the initial slash if necessary
                    if image_location and image_location.startswith('/'):
                        image_location = image_location[1:]
                    logger.debug(f"[{request_id}] Using relative path for image location: {image_location}")

                # If there is a Path, we use it as a URL image
                if "fileContent" in asset_data and "path" in asset_data["fileContent"]:
                    image_url = asset_data["fileContent"]["path"]
                    # Extract a relative path if the URL contains the domain
                    if "asset.1min.ai/" in image_url:
                        image_url = image_url.split('asset.1min.ai/', 1)[1]
                    # Remove the initial slash if necessary
                    if image_url.startswith('/'):
                        image_url = image_url[1:]
                    logger.debug(f"[{request_id}] Using relative path for image: {image_url}")

                if not (image_id or image_url or image_location):
                    logger.error(f"[{request_id}] Failed to extract image information from response")
                    continue  # We try the next model

                # We form Payload for image variation
                # We determine which model to use
                if model.startswith("midjourney"):
                    # Check if the URL contains the Asset.1Min.Ai domain
                    if image_url and "asset.1min.ai/" in image_url:
                        # We extract only the relative path from the URL
                        relative_image_url = image_url.split('asset.1min.ai/', 1)[1]
                        # Remove the initial slash if it is
                        if relative_image_url.startswith('/'):
                            relative_image_url = relative_image_url[1:]
                        logger.info(f"[{request_id}] Extracted relative URL for Midjourney: {relative_image_url}")
                    else:
                        relative_image_url = image_url if image_url else image_location
                        if relative_image_url and relative_image_url.startswith('/'):
                            relative_image_url = relative_image_url[1:]
                    
                    # For Midjourney
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": model,
                        "promptObject": {
                            "imageUrl": relative_image_url if relative_image_url else image_url if image_url else image_location,
                            "mode": mode or "fast",
                            "n": 4,
                            "isNiji6": False,
                            "aspect_width": aspect_width or 1,
                            "aspect_height": aspect_height or 1,
                            "maintainModeration": True
                        }
                    }
                elif model == "dall-e-2":
                    # For Dall-E 2
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "dall-e-2",
                        "promptObject": {
                            "imageUrl": relative_image_url if relative_image_url else image_url if image_url else image_location,
                            "n": 1,
                            "size": "1024x1024"
                        }
                    }
                elif model == "clipdrop":
                    # For Clipdrop (Stable Diffusion)
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "clipdrop",
                        "promptObject": {
                            "imageUrl": relative_image_url if relative_image_url else image_url if image_url else image_location,
                        }
                    }
                else:
                    # For all other models, we use minimal parameters
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": model,
                        "promptObject": {
                            "imageUrl": relative_image_url if relative_image_url else image_url if image_url else image_location,
                            "n": int(n)
                        }
                    }

                # Remove the initial slash in Imageurl if it is
                if "imageUrl" in payload["promptObject"] and payload["promptObject"]["imageUrl"] and isinstance(
                        payload["promptObject"]["imageUrl"], str) and payload["promptObject"]["imageUrl"].startswith(
                    '/'):
                    payload["promptObject"]["imageUrl"] = payload["promptObject"]["imageUrl"][1:]
                    logger.debug(
                        f"[{request_id}] Removed leading slash from imageUrl: {payload['promptObject']['imageUrl']}")

                # For VIP users, add Credit to the request
                if api_key.startswith("vip-"):
                    payload["credits"] = 90000  # Standard number of loans for VIP

                # Detailed Payload logistics for debugging
                logger.info(f"[{request_id}] {model} variation payload: {json.dumps(payload, indent=2)}")

                # Using Timeout for all models (10 minutes)
                timeout = MIDJOURNEY_TIMEOUT

                logger.debug(f"[{request_id}] Sending variation request to {ONE_MIN_API_URL}")

                # We send a request to create a variation
                variation_response = api_request(
                    "POST",
                    f"{ONE_MIN_API_URL}",
                    headers={"API-KEY": api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=timeout
                )

                if variation_response.status_code != 200:
                    # We process the 504 error for Midjourney in a special way
                    if variation_response.status_code == 504 and model.startswith("midjourney"):
                        logger.error(
                            f"[{request_id}] Received a 504 Gateway Timeout for Midjourney variations. Returning the error to the client.")
                        return (
                            jsonify(
                                {"error": "Gateway Timeout (504) occurred while processing image variation request."}),
                            504,
                        )
                    # For other errors, we continue to try the next model
                    logger.error(
                        f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
                    continue

                # We process the answer and form the result
                variation_data = variation_response.json()
                # Add a detailed log for myidjourney model
                if model.startswith("midjourney"):
                    logger.info(f"[{request_id}] Full Midjourney variation response: {json.dumps(variation_data, indent=2)}")
                logger.debug(f"[{request_id}] Variation response: {variation_data}")

                # We extract the URL variations - initialize an empty array before searching
                variation_urls = []
                # We are trying to find URL variations in the answer - various structures for different models
                if "aiRecord" in variation_data and "aiRecordDetail" in variation_data["aiRecord"]:
                    record_detail = variation_data["aiRecord"]["aiRecordDetail"]
                    if "resultObject" in record_detail:
                        result = record_detail["resultObject"]
                        if isinstance(result, list):
                            variation_urls = result
                        elif isinstance(result, str):
                            variation_urls = [result]

                # An alternative search path
                if not variation_urls and "resultObject" in variation_data:
                    result = variation_data["resultObject"]
                    if isinstance(result, list):
                        variation_urls = result
                    elif isinstance(result, str):
                        variation_urls = [result]

                # Search in Data.URL for Dall-E 2
                if not variation_urls and "data" in variation_data and isinstance(variation_data["data"], list):
                    for item in variation_data["data"]:
                        if "url" in item:
                            variation_urls.append(item["url"])

                if not variation_urls:
                    logger.error(f"[{request_id}] No variation URLs found in response with model {model}")
                    continue  # We try the next model

                # Successfully received variations, we leave the cycle
                logger.info(f"[{request_id}] Successfully generated variations with model {model}")
                break

        except Exception as e:
            logger.error(f"[{request_id}] Exception during variation request with model {model}: {str(e)}")
            continue  # We try the next model

    # Clean the temporary file
    try:
        os.unlink(temp_file.name)
    except:
        pass

    # We check if you managed to get variations from any of the models
    if not variation_urls:
        session.close()
        return jsonify({"error": "Failed to create image variations with any available model"}), 500

    # We form complete URL for variations
    full_variation_urls = []
    asset_host = "https://asset.1min.ai"

    for url in variation_urls:
        if not url:
            continue

        # We save the relative path for the API, but create a full URL for display
        relative_url = url
        # If the URL contains a domain, we extract a relative path
        if "asset.1min.ai/" in url:
            relative_url = url.split('asset.1min.ai/', 1)[1]
            # Remove the initial slash if it is
            if relative_url.startswith('/'):
                relative_url = relative_url[1:]
        # If the URL is already without a domain, but starts with the slashus, we remove the slash
        elif url.startswith('/'):
            relative_url = url[1:]

        # Create a full URL to display the user
        if not url.startswith("http"):
            if url.startswith("/"):
                full_url = f"{asset_host}{url}"
            else:
                full_url = f"{asset_host}/{url}"
        else:
            full_url = url

        # We keep the relative path and full URL
        full_variation_urls.append({
            "relative_path": relative_url,
            "full_url": full_url
        })

    # We form an answer in Openai format
    openai_data = []
    for url_data in full_variation_urls:
        # We use the relative path for the API
        openai_data.append({"url": url_data["relative_path"]})

    openai_response = {
        "created": int(time.time()),
        "data": openai_data,
    }

    # Add the text with variation buttons for Markdown Object
    markdown_text = ""
    if len(full_variation_urls) == 1:
        # We use the full URL to display
        markdown_text = f"![Variation]({full_variation_urls[0]['full_url']}) `[_V1_]`"
        # Add a hint to create variations
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** and send it (paste) in the next **prompt**"
    else:
        # We form a text with images and buttons of variations on one line
        image_lines = []

        for i, url_data in enumerate(full_variation_urls):
            # We use the full URL to display
            image_lines.append(f"![Variation {i + 1}]({url_data['full_url']}) `[_V{i + 1}_]`")

        # Combine lines with a new line between them
        markdown_text = "\n".join(image_lines)
        # Add a hint to create variations
        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**"

    openai_response["choices"] = [
        {
            "message": {
                "role": "assistant",
                "content": markdown_text
            },
            "index": 0,
            "finish_reason": "stop"
        }
    ]

    session.close()
    logger.info(
        f"[{request_id}] Successfully generated {len(openai_data)} image variations using model {current_model}")
    return jsonify(openai_response), 200

# Вспомогательные функции для изображений
def parse_aspect_ratio(prompt, model, request_data, request_id=None):
    """
    Extracts the ratio of the parties from the request or prompt and checks its validity

    Args:
        PROMPT (STR): Request text
        Model (str): the name of the image generation model
        Request_Data (DICT): Request data
        Request_id (Str, Optional): ID Request for Logging

    Returns:
        tuple: (modified Prompt, parties ratio, image size, error message, mode)
    """
    # Default values
    aspect_ratio = None
    size = request_data.get("size", "1024x1024")
    ar_error = None
    mode = None

    # We are looking for the parameters of the mode in the text
    mode_match = re.search(r'(--|\u2014)(fast|relax)\s*', prompt)
    if mode_match:
        mode = mode_match.group(2)
        # We delete the parameter of the process from the prompt
        prompt = re.sub(r'(--|\u2014)(fast|relax)\s*', '', prompt).strip()
        logger.debug(f"[{request_id}] Extracted mode from prompt: {mode}")

    # We are trying to extract the ratio of the parties from Prompt
    ar_match = re.search(r'(--|\u2014)ar\s+(\d+):(\d+)', prompt)
    if ar_match:
        width = int(ar_match.group(2))
        height = int(ar_match.group(3))

        # We check that the ratio does not exceed 2: 1 or 1: 2
        if max(width, height) / min(width, height) > 2:
            ar_error = "Aspect ratio cannot exceed 2:1 or 1:2"
            logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error, mode

        # We check that the values ​​in the permissible range
        if width < 1 or width > 10000 or height < 1 or height > 10000:
            ar_error = "Aspect ratio values must be between 1 and 10000"
            logger.error(f"[{request_id}] Invalid aspect ratio values: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error, mode

        # Install the ratio of the parties
        aspect_ratio = f"{width}:{height}"

        # We delete the parameter from prompt
        prompt = re.sub(r'(--|\u2014)ar\s+\d+:\d+\s*', '', prompt).strip()

        logger.debug(f"[{request_id}] Extracted aspect ratio: {aspect_ratio}")

    # If there is no ratio in Prompta, we check in the request
    elif "aspect_ratio" in request_data:
        aspect_ratio = request_data.get("aspect_ratio")

        # We check that the ratio in the correct format
        if not re.match(r'^\d+:\d+$', aspect_ratio):
            ar_error = "Aspect ratio must be in format width:height"
            logger.error(f"[{request_id}] Invalid aspect ratio format: {aspect_ratio} - {ar_error}")
            return prompt, None, size, ar_error, mode

        width, height = map(int, aspect_ratio.split(':'))

        # We check that the ratio does not exceed 2: 1 or 1: 2
        if max(width, height) / min(width, height) > 2:
            ar_error = "Aspect ratio cannot exceed 2:1 or 1:2"
            logger.error(f"[{request_id}] Invalid aspect ratio: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error, mode

        # We check that the values ​​in the permissible range
        if width < 1 or width > 10000 or height < 1 or height > 10000:
            ar_error = "Aspect ratio values must be between 1 and 10000"
            logger.error(f"[{request_id}] Invalid aspect ratio values: {width}:{height} - {ar_error}")
            return prompt, None, size, ar_error, mode

        logger.debug(f"[{request_id}] Using aspect ratio from request: {aspect_ratio}")

    # We delete all other possible modifiers of parameters
    # Remove negative promptists (-no or –no)
    prompt = re.sub(r'(--|\u2014)no\s+.*?(?=(--|\u2014)|$)', '', prompt).strip()

    # For models Dall-E 3, set the corresponding dimensions
    if model == "dall-e-3" and aspect_ratio:
        width, height = map(int, aspect_ratio.split(':'))

        # We round to the nearest permissible ratio for Dall-E 3
        if abs(width / height - 1) < 0.1:  # square
            size = "1024x1024"
            aspect_ratio = "square"
        elif width > height:  # Album orientation
            size = "1792x1024"
            aspect_ratio = "landscape"
        else:  # Portrait orientation
            size = "1024x1792"
            aspect_ratio = "portrait"

        logger.debug(f"[{request_id}] Adjusted size for DALL-E 3: {size}, aspect_ratio: {aspect_ratio}")

    # For Leonardo models, we set the corresponding dimensions based on the ratio of the parties
    elif (model in [
        "6b645e3a-d64f-4341-a6d8-7a3690fbf042", "phoenix",  # Leonardo.ai - Phoenix
        "b24e16ff-06e3-43eb-8d33-4416c2d75876", "lightning-xl",  # Leonardo.ai - Lightning XL
        "5c232a9e-9061-4777-980a-ddc8e65647c6", "vision-xl",  # Leonardo.ai - Vision XL
        "e71a1c2f-4f80-4800-934f-2c68979d8cc8", "anime-xl",  # Leonardo.ai - Anime XL
        "1e60896f-3c26-4296-8ecc-53e2afecc132", "diffusion-xl",  # Leonardo.ai - Diffusion XL
        "aa77f04e-3eec-4034-9c07-d0f619684628", "kino-xl",  # Leonardo.ai - Kino XL
        "2067ae52-33fd-4a82-bb92-c2c55e7d2786", "albedo-base-xl"  # Leonardo.ai - Albedo Base XL
    ]) and aspect_ratio:
        # Determine the size based on the ratio of the parties
        if aspect_ratio == "1:1":
            size = LEONARDO_SIZES["1:1"]  # "1024x1024"
        elif aspect_ratio == "4:3":
            size = LEONARDO_SIZES["4:3"]  # "1024x768"
        elif aspect_ratio == "3:4":
            size = LEONARDO_SIZES["3:4"]  # "768x1024"
        # For other ratios, we round to the nearest supported
        else:
            width, height = map(int, aspect_ratio.split(':'))
            ratio = width / height

            if abs(ratio - 1) < 0.1:  # Close to 1: 1
                size = LEONARDO_SIZES["1:1"]  # "1024x1024"
                aspect_ratio = "1:1"
            elif ratio > 1:  # The width is greater than the height (album orientation)
                size = LEONARDO_SIZES["4:3"]  # "1024x768"
                aspect_ratio = "4:3"
            else:  # The height is greater than the width (portrait orientation)
                size = LEONARDO_SIZES["3:4"]  # "768x1024"
                aspect_ratio = "3:4"

        logger.debug(f"[{request_id}] Adjusted size for Leonardo model: {size}, aspect_ratio: {aspect_ratio}")

    return prompt, aspect_ratio, size, ar_error, mode

def retry_image_upload(image_url, api_key, request_id=None):
    """Uploads an image with repeated attempts, returns a direct link to it"""
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Uploading image: {image_url}")

    # We create a new session for this request
    session = create_session()
    temp_file_path = None

    try:
        # We load the image
        if image_url.startswith(("http://", "https://")):
            # URL loading
            logger.debug(f"[{request_id}] Fetching image from URL: {image_url}")
            response = session.get(image_url, stream=True)
            response.raise_for_status()
            image_data = response.content
        else:
            # Decoding Base64
            logger.debug(f"[{request_id}] Decoding base64 image")
            image_data = base64.b64decode(image_url.split(",")[1])

        # Check the file size
        if len(image_data) == 0:
            logger.error(f"[{request_id}] Empty image data")
            return None

        # Create a temporary file
        temp_file_path = safe_temp_file("image", request_id)

        with open(temp_file_path, "wb") as f:
            f.write(image_data)

        # Check that the file is not empty
        if os.path.getsize(temp_file_path) == 0:
            logger.error(f"[{request_id}] Empty image file created: {temp_file_path}")
            return None

        # We load to the server
        try:
            with open(temp_file_path, "rb") as f:
                upload_response = session.post(
                    ONE_MIN_ASSET_URL,
                    headers={"API-KEY": api_key},
                    files={
                        "asset": (
                            os.path.basename(image_url),
                            f,
                            (
                                "image/webp"
                                if image_url.endswith(".webp")
                                else "image/jpeg"
                            ),
                        )
                    },
                )

                if upload_response.status_code != 200:
                    logger.error(
                        f"[{request_id}] Upload failed with status {upload_response.status_code}: {upload_response.text}"
                    )
                    return None

                # We get URL images
                upload_data = upload_response.json()
                if isinstance(upload_data, str):
                    try:
                        upload_data = json.loads(upload_data)
                    except:
                        logger.error(
                            f"[{request_id}] Failed to parse upload response: {upload_data}"
                        )
                        return None

                logger.debug(f"[{request_id}] Upload response: {upload_data}")

                # We get the path to the file from FileContent
                if (
                        "fileContent" in upload_data
                        and "path" in upload_data["fileContent"]
                ):
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
        # Close the session
        session.close()
        # We delete a temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"[{request_id}] Removed temp file: {temp_file_path}")
            except Exception as e:
                logger.warning(
                    f"[{request_id}] Failed to remove temp file {temp_file_path}: {str(e)}"
                )

def create_image_variations(image_url, user_model, n, aspect_width=None, aspect_height=None, mode=None,
                            request_id=None):
    """
    Creates variations based on the original image, taking into account the specifics of each model.
    """
    # Initialize the URL list in front of the cycle
    variation_urls = []
    current_model = None

    # We use a temporary ID request if it was not provided
    if request_id is None:
        request_id = str(uuid.uuid4())

    # We get saved generation parameters
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

    # We use saved parameters if they are available
    if generation_params:
        # We take Aspect_width and Aspect_Height from saved parameters if they are
        if "aspect_width" in generation_params and "aspect_height" in generation_params:
            aspect_width = generation_params.get("aspect_width")
            aspect_height = generation_params.get("aspect_height")
            logger.debug(f"[{request_id}] Using saved aspect ratio: {aspect_width}:{aspect_height}")

        # We take the mode of saved parameters if it is
        if "mode" in generation_params:
            mode = generation_params.get("mode")
            logger.debug(f"[{request_id}] Using saved mode: {mode}")

    # We determine the list of models for variations
    variation_models = []
    if user_model in VARIATION_SUPPORTED_MODELS:
        variation_models.append(user_model)
    variation_models.extend([m for m in ["midjourney_6_1", "midjourney", "clipdrop", "dall-e-2"] if m != user_model])
    variation_models = list(dict.fromkeys(variation_models))

    logger.info(f"[{request_id}] Trying image variations with models: {variation_models}")

    # Create a session to download the image
    session = create_session()

    try:
        # We load the image
        image_response = session.get(image_url, stream=True, timeout=60)
        if image_response.status_code != 200:
            logger.error(f"[{request_id}] Failed to download image: {image_response.status_code}")
            return jsonify({"error": "Failed to download image"}), 500

        # We try each model in turn
        for model in variation_models:
            current_model = model
            logger.info(f"[{request_id}] Trying model: {model} for image variations")

            try:
                # Determine the MIME-type image based on the contents or url
                content_type = "image/png"  # By default
                if "content-type" in image_response.headers:
                    content_type = image_response.headers["content-type"]
                elif image_url.lower().endswith(".webp"):
                    content_type = "image/webp"
                elif image_url.lower().endswith(".jpg") or image_url.lower().endswith(".jpeg"):
                    content_type = "image/jpeg"
                elif image_url.lower().endswith(".gif"):
                    content_type = "image/gif"

                # Determine the appropriate extension for the file
                ext = "png"
                if "webp" in content_type:
                    ext = "webp"
                elif "jpeg" in content_type or "jpg" in content_type:
                    ext = "jpg"
                elif "gif" in content_type:
                    ext = "gif"

                logger.debug(f"[{request_id}] Detected image type: {content_type}, extension: {ext}")

                # We load the image to the server with the correct MIME type
                files = {"asset": (f"variation.{ext}", image_response.content, content_type)}
                upload_response = session.post(ONE_MIN_ASSET_URL, files=files, headers=headers)

                if upload_response.status_code != 200:
                    logger.error(f"[{request_id}] Image upload failed: {upload_response.status_code}")
                    continue

                upload_data = upload_response.json()
                logger.debug(f"[{request_id}] Asset upload response: {upload_data}")

                # We get the way to the loaded image
                image_path = None
                if "fileContent" in upload_data and "path" in upload_data["fileContent"]:
                    image_path = upload_data["fileContent"]["path"]
                    # We remove the initial slash if it is
                    if image_path.startswith('/'):
                        image_path = image_path[1:]
                    logger.debug(f"[{request_id}] Using relative path for variation: {image_path}")
                else:
                    logger.error(f"[{request_id}] Could not extract image path from upload response")
                    continue

                # We form Payload depending on the model
                if model in ["midjourney_6_1", "midjourney"]:
                    # For Midjourney
                    # Add the URL transformation
                    if image_url and isinstance(image_url, str) and 'asset.1min.ai/' in image_url:
                        image_url = image_url.split('asset.1min.ai/', 1)[1]
                        logger.debug(f"[{request_id}] Extracted path from image_url: {image_url}")

                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": model,
                        "promptObject": {
                            "imageUrl": image_url if image_url else image_location,
                            "mode": mode or request_data.get("mode", "fast"),  # We use the mode from the prompt
                            "n": 4,
                            "isNiji6": False,
                            "aspect_width": aspect_width or 1,
                            "aspect_height": aspect_height or 1,
                            "maintainModeration": True
                        }
                    }
                    # Detailed logistics for Midjourney
                    logger.info(f"[{request_id}] Midjourney variation payload:")
                    logger.info(f"[{request_id}] promptObject: {json.dumps(payload['promptObject'], indent=2)}")
                elif model == "dall-e-2":
                    # For Dall-E 2
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

                    # We send a request through the main API URL
                    variation_response = api_request(
                        "POST",
                        ONE_MIN_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=300
                    )

                    if variation_response.status_code != 200:
                        logger.error(
                            f"[{request_id}] DALL-E 2 variation failed: {variation_response.status_code}, {variation_response.text}")
                        continue

                    # We process the answer
                    variation_data = variation_response.json()

                    # We extract the URL from the answer
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
                        logger.info(
                            f"[{request_id}] Successfully created {len(variation_urls)} variations with DALL-E 2")
                        break
                    else:
                        logger.warning(f"[{request_id}] No variation URLs found in DALL-E 2 response")
                elif model == "clipdrop":
                    # For Clipdrop
                    payload = {
                        "type": "IMAGE_VARIATOR",
                        "model": "clipdrop",
                        "promptObject": {
                            "imageUrl": image_path,
                            "n": n
                        }
                    }
                    logger.info(f"[{request_id}] Clipdrop variation payload: {json.dumps(payload, indent=2)}")

                    # We send a request through the main API URL
                    variation_response = api_request(
                        "POST",
                        ONE_MIN_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=300
                    )

                    if variation_response.status_code != 200:
                        logger.error(
                            f"[{request_id}] Clipdrop variation failed: {variation_response.status_code}, {variation_response.text}")
                        continue

                    # We process the answer
                    variation_data = variation_response.json()

                    # We extract the URL from the answer
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
                        logger.info(
                            f"[{request_id}] Successfully created {len(variation_urls)} variations with Clipdrop")
                        break
                    else:
                        logger.warning(f"[{request_id}] No variation URLs found in Clipdrop response")

                logger.debug(f"[{request_id}] Sending variation request to URL: {ONE_MIN_API_URL}")
                logger.debug(f"[{request_id}] Using headers: {json.dumps(headers)}")

                # We send a request to create a variation
                timeout = MIDJOURNEY_TIMEOUT if model.startswith("midjourney") else DEFAULT_TIMEOUT
                logger.debug(f"Using extended timeout for Midjourney: {timeout}s")

                variation_response = api_request(
                    "POST",
                    ONE_MIN_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )

                if variation_response.status_code != 200:
                    logger.error(
                        f"[{request_id}] Variation request with model {model} failed: {variation_response.status_code} - {variation_response.text}")
                    continue

                # We process the answer
                variation_data = variation_response.json()

                # We extract the URL variations from the answer
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

        # If you could not create variations with any model
        if not variation_urls:
            logger.error(f"[{request_id}] Failed to create variations with any available model")
            return jsonify({"error": "Failed to create image variations with any available model"}), 500

        # We form an answer
        openai_response = {
            "created": int(time.time()),
            "data": []
        }

        for url in variation_urls:
            openai_data = {
                "url": url
            }
            openai_response["data"].append(openai_data)

        # We form a markdown text with a hint
        text_lines = []
        for i, url in enumerate(variation_urls, 1):
            text_lines.append(f"Image {i} ({url}) [_V{i}_]")
        text_lines.append(
            "\n> To generate **variants** of **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**")

        text_response = "\n".join(text_lines)

        openai_response["choices"] = [{
            "message": {
                "role": "assistant",
                "content": text_response
            },
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
