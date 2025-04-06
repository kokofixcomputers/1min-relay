# Маршруты для текстовых моделей
# Импортируем только необходимые модули
from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import ERROR_HANDLER, handle_options_request, set_response_headers, create_session, api_request, safe_temp_file, calculate_token
from . import app, limiter, IMAGE_CACHE, MAX_CACHE_SIZE  # Импортируем app, limiter и IMAGE_CACHE из модуля routes

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
    # Dynamically create the list of models with additional fields
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

    if not request.json:
        return jsonify({"error": "Invalid request format"}), 400

    # We extract information from the request
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not api_key:
        logger.error(f"[{request_id}] No API key provided")
        return jsonify({"error": "API key required"}), 401

    try:
        # Build Payload for request
        request_data = request.json.copy()

        # We get and normalize the model
        model = request_data.get("model", "").strip()
        logger.info(f"[{request_id}] Using model: {model}")

        # We check the support of the web post for the model
        capabilities = get_model_capabilities(model)

        # We check if the web post is requested through Openai tools
        web_search_requested = False
        tools = request_data.get("tools", [])
        for tool in tools:
            if tool.get("type") == "retrieval":
                web_search_requested = True
                logger.debug(f"[{request_id}] Web search requested via retrieval tool")
                break

        # Check the presence of the Web_Search parameter
        if not web_search_requested and request_data.get("web_search", False):
            web_search_requested = True
            logger.debug(f"[{request_id}] Web search requested via web_search parameter")

        # Add a clear web_search parameter if you are requested and supported by the model
        if web_search_requested:
            if capabilities["retrieval"]:
                request_data["web_search"] = True
                request_data["num_of_site"] = request_data.get("num_of_site", 1)
                request_data["max_word"] = request_data.get("max_word", 1000)
                logger.info(f"[{request_id}] Web search enabled for model {model}")
            else:
                logger.warning(f"[{request_id}] Model {model} does not support web search, ignoring request")

        # We extract the contents of the last message for possible generation of images
        messages = request_data.get("messages", [])
        prompt_text = ""
        if messages and len(messages) > 0:
            last_message = messages[-1]
            if last_message.get("role") == "user":
                content = last_message.get("content", "")
                if isinstance(content, str):
                    prompt_text = content
                elif isinstance(content, list):
                    # Collect all the text parts of the contents
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            prompt_text += item["text"] + " "
                    prompt_text = prompt_text.strip()

        # We check whether the request contains the variation of the image
        variation_match = None
        if prompt_text:
            # We are looking for the format of old teams /v1- /v4
            old_variation_match = re.search(r'/v([1-4])\s+(https?://[^\s]+)', prompt_text)
            # We are looking for a format with square brackets [_v1 _]-[_ v4_]
            square_variation_match = re.search(r'\[_V([1-4])_\]', prompt_text)
            # We are looking for a new format with monoshyrin text `[_V1_]` -` [_V4_] `
            mono_variation_match = re.search(r'`\[_V([1-4])_\]`', prompt_text)

            # If a monoshyrin format is found, we check if there is a URL dialogue in the history
            if mono_variation_match and request_data.get("messages"):
                variation_number = int(mono_variation_match.group(1))
                logger.debug(f"[{request_id}] Found monospace format variation command: {variation_number}")

                # Looking for the necessary URL in previous messages of the assistant
                image_url = None
                for msg in reversed(request_data.get("messages", [])):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        # Looking for all URL images in the content of the assistant message
                        content = msg.get("content", "")
                        # We use a more specific regular expression to search for images with the corresponding numbers
                        image_urls = []
                        # First, we are looking for all URL images in standard Markdown format
                        url_matches = re.findall(r'!\[(?:Variation\s*(\d+)|[^]]*)\]\((https?://[^\s)]+)', content)
                        
                        # We convert the results to the list, taking into account variation rooms
                        for match in url_matches:
                            # If there is a variation number, we use it for indexing
                            variation_num = None
                            if match[0]:  # If the variation number was found
                                try:
                                    variation_num = int(match[0].strip())
                                except ValueError:
                                    pass
                            
                            # URL always the second element of the group
                            url = match[1]
                            
                            # Add to the list with the corresponding index or simply add to the end
                            if variation_num and 0 < variation_num <= 10:  # Limit up to 10 variations maximum
                                # We expand the list to the desired length, if necessary
                                while len(image_urls) < variation_num:
                                    image_urls.append(None)
                                image_urls[variation_num-1] = url
                            else:
                                image_urls.append(url)
                        
                        # We delete all None values ​​from the list
                        image_urls = [url for url in image_urls if url is not None]
                        
                        if image_urls:
                            # Check the URL number
                            if len(image_urls) >= variation_number:
                                # We take the URL corresponding to the requested number
                                image_url = image_urls[variation_number - 1]
                                logger.debug(
                                    f"[{request_id}] Found image URL #{variation_number} in assistant message: {image_url}")
                                break
                            else:
                                # Not enough URL for the requested number, we take the first
                                image_url = image_urls[0]
                                logger.warning(
                                    f"[{request_id}] Requested variation #{variation_number} but only found {len(image_urls)} URLs. Using first URL: {image_url}")
                                break

                if image_url:
                    variation_match = mono_variation_match
                    logger.info(
                        f"[{request_id}] Detected monospace variation command: {variation_number} for URL: {image_url}")
            # If a format with square brackets is found, we check if there is a URL dialogue in the history
            elif square_variation_match and request_data.get("messages"):
                variation_number = int(square_variation_match.group(1))
                logger.debug(f"[{request_id}] Found square bracket format variation command: {variation_number}")

                # Looking for the necessary URL in previous messages of the assistant
                image_url = None
                for msg in reversed(request_data.get("messages", [])):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        # Looking for all URL images in the content of the assistant message
                        content = msg.get("content", "")
                        url_matches = re.findall(r'!\[.*?\]\((https?://[^\s)]+)', content)

                        if url_matches:
                            # Check the number of URL found
                            if len(url_matches) >= variation_number:
                                # We take the URL corresponding to the requested number
                                image_url = url_matches[variation_number - 1]
                                logger.debug(
                                    f"[{request_id}] Found image URL #{variation_number} in assistant message: {image_url}")
                                break
                            else:
                                # Not enough URL for the requested number, we take the first
                                image_url = url_matches[0]
                                logger.warning(
                                    f"[{request_id}] Requested variation #{variation_number} but only found {len(url_matches)} URLs. Using first URL: {image_url}")
                                break

                if image_url:
                    variation_match = square_variation_match
                    logger.info(
                        f"[{request_id}] Detected square bracket variation command: {variation_number} for URL: {image_url}")
            # If the old format is found, we use it
            elif old_variation_match:
                variation_match = old_variation_match
                variation_number = old_variation_match.group(1)
                image_url = old_variation_match.group(2)
                logger.info(
                    f"[{request_id}] Detected old format variation command: {variation_number} for URL: {image_url}")

        if variation_match:
            # We process the variation of the image
            try:
                # We check what type of variation was discovered
                if variation_match == mono_variation_match or variation_match == square_variation_match:
                    # URL has already been obtained above in the search process
                    variation_number = variation_match.group(1)
                else:
                    # For the old format, we extract the URL directly from the team
                    variation_number = variation_match.group(1)
                    image_url = variation_match.group(2)

                logger.info(f"[{request_id}] Processing variation for image: {image_url}")

                # For Midjourney models, add a direct call of the API without downloading the image
                if model.startswith("midjourney") and "asset.1min.ai" in image_url:
                    # We extract a relative path from the URL
                    path_match = re.search(r'(?:asset\.1min\.ai)/?(images/[^?#]+)', image_url)
                    if path_match:
                        relative_path = path_match.group(1)
                        logger.info(f"[{request_id}] Detected Midjourney variation with relative path: {relative_path}")
                        
                        # We get the saved generation parameters from Memcached by Request_id
                        saved_params = None
                        try:
                            # We extract image_id from the image path for searching for parameters
                            image_id_match = re.search(r'images/(\d+_\d+_\d+_\d+_\d+_\d+|\w+\d+)\.png', relative_path)
                            if image_id_match:
                                image_id = image_id_match.group(1)
                                logger.info(f"[{request_id}] Extracted image_id for variation: {image_id}")
                                gen_params_key = f"gen_params:{image_id}"
                                logger.info(f"[{request_id}] Looking for generation parameters with key: {gen_params_key}")
                                
                                # Check the presence of parameters in Memory_Storage directly
                                if gen_params_key in MEMORY_STORAGE:
                                    stored_value = MEMORY_STORAGE[gen_params_key]
                                    logger.info(f"[{request_id}] Found in MEMORY_STORAGE (type: {type(stored_value)}): {stored_value}")
                                    
                                    # If the value is a line, we try to convert it into a python dictionary
                                    if isinstance(stored_value, str):
                                        try:
                                            saved_params = json.loads(stored_value)
                                            logger.info(f"[{request_id}] Successfully parsed JSON string to dict")
                                        except Exception as e:
                                            logger.error(f"[{request_id}] Failed to parse JSON string: {e}")
                                            saved_params = stored_value
                                    else:
                                        saved_params = stored_value
                                        
                                    logger.info(f"[{request_id}] Using parameters directly from MEMORY_STORAGE (type: {type(saved_params)}): {saved_params}")
                                else:
                                    # If you are not found in Memory_Storage, we try it through Safe_memcache_oporation
                                    logger.info(f"[{request_id}] Not found in MEMORY_STORAGE, trying safe_memcached_operation")
                                    params_json = safe_memcached_operation('get', gen_params_key)
                                    if params_json:
                                        logger.info(f"[{request_id}] Retrieved parameters for image {image_id}: {params_json}")
                                        if isinstance(params_json, str):
                                            try:
                                                saved_params = json.loads(params_json)
                                            except:
                                                saved_params = params_json
                                        elif isinstance(params_json, bytes):
                                            try:
                                                saved_params = json.loads(params_json.decode('utf-8'))
                                            except:
                                                saved_params = params_json.decode('utf-8')
                                        else:
                                            saved_params = params_json
                                        logger.info(f"[{request_id}] Retrieved generation parameters for image {image_id}: {saved_params}")
                                    else:
                                        logger.info(f"[{request_id}] No parameters found in storage for key {gen_params_key}")
                        except Exception as e:
                            logger.error(f"[{request_id}] Error retrieving generation parameters: {str(e)}")
                        
                        # We form Payload for variation
                        payload = {
                            "type": "IMAGE_VARIATOR",
                            "model": model,
                            "promptObject": {
                                "imageUrl": relative_path,
                                "mode": "fast",  # Default Fast mode
                                "n": 4,
                                "isNiji6": False,
                                "aspect_width": 1,  # By default 1: 1
                                "aspect_height": 1,  # By default 1: 1
                                "maintainModeration": True
                            }
                        }
                        
                        # We use parameters from Memcache if they are available
                        if saved_params:
                            logger.info(f"[{request_id}] Using saved parameters from original generation: {saved_params}")
                            # We will transfer all the saved parameters
                            for param in ["mode", "aspect_width", "aspect_height", "isNiji6", "maintainModeration"]:
                                if param in saved_params:
                                    old_value = payload["promptObject"].get(param)
                                    payload["promptObject"][param] = saved_params[param]
                                    logger.info(f"[{request_id}] Changed parameter {param} from {old_value} to {saved_params[param]}")
                        else:
                            logger.info(f"[{request_id}] No saved parameters found, using default ratio 1:1 for Midjourney variations")
                            # We use the ratio of 1: 1
                            payload["promptObject"]["aspect_width"] = 1
                            payload["promptObject"]["aspect_height"] = 1
                        
                        # We send a request for variation directly
                        logger.info(f"[{request_id}] Sending direct Midjourney variation request: {json.dumps(payload)}")
                        
                        try:
                            variation_response = api_request(
                                "POST",
                                f"{ONE_MIN_API_URL}",
                                headers={"API-KEY": api_key, "Content-Type": "application/json"},
                                json=payload,
                                timeout=MIDJOURNEY_TIMEOUT
                            )
                            
                            if variation_response.status_code == 200:
                                # We process a successful answer
                                variation_data = variation_response.json()
                                logger.info(f"[{request_id}] Received Midjourney variation response: {json.dumps(variation_data)}")
                                
                                # We extract the URL variations
                                variation_urls = []
                                
                                # Midjourney structure structure
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
                                
                                if variation_urls:
                                    logger.info(f"[{request_id}] Found {len(variation_urls)} variation URLs")
                                    
                                    # We form full URLs for display
                                    full_variation_urls = []
                                    asset_host = "https://asset.1min.ai"
                                    
                                    for url in variation_urls:
                                        # Create a full URL to display
                                        if not url.startswith("http"):
                                            full_url = f"{asset_host}/{url}"
                                        else:
                                            full_url = url
                                        
                                        full_variation_urls.append(full_url)
                                    
                                    # We form a response in Markdown format
                                    markdown_text = ""
                                    if len(full_variation_urls) == 1:
                                        markdown_text = f"![Variation]({full_variation_urls[0]}) `[_V1_]`"
                                        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** and send it (paste) in the next **prompt**"
                                    else:
                                        image_lines = []
                                        for i, url in enumerate(full_variation_urls):
                                            image_lines.append(f"![Variation {i + 1}]({url}) `[_V{i + 1}_]`")
                                        
                                        markdown_text = "\n".join(image_lines)
                                        markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]** - **[_V4_]** and send it (paste) in the next **prompt**"
                                    
                                    # We form an answer in Openai format
                                    openai_response = {
                                        "id": f"chatcmpl-{uuid.uuid4()}",
                                        "object": "chat.completion",
                                        "created": int(time.time()),
                                        "model": model,
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
                                        "usage": {
                                            "prompt_tokens": 0,
                                            "completion_tokens": 0,
                                            "total_tokens": 0
                                        }
                                    }
                                    
                                    return jsonify(openai_response), 200
                                else:
                                    logger.error(f"[{request_id}] No variation URLs found in response")
                            else:
                                logger.error(f"[{request_id}] Direct variation request failed: {variation_response.status_code} - {variation_response.text}")
                                # When the Gateway Timeout (504) error, we return the error immediately, and do not continue to process
                                if variation_response.status_code == 504:
                                    logger.error(f"[{request_id}] Midjourney API timeout (504). Returning error to client instead of fallback.")
                                    return jsonify({
                                        "error": "Gateway Timeout (504) occurred while processing image variation request. Try again later."
                                    }), 504
                                # With an error with the ratio of the parties (409), we also return the error
                                elif variation_response.status_code == 409:
                                    error_message = "Error creating image variation"
                                    # Trying to extract an error from an answer
                                    try:
                                        error_json = variation_response.json()
                                        if "message" in error_json:
                                            error_message = error_json["message"]
                                    except:
                                        pass
                                    logger.error(f"[{request_id}] Midjourney API error (409): {error_message}")
                                    return jsonify({
                                        "error": f"Failed to create image variation: {error_message}"
                                    }), 409
                        except Exception as e:
                            logger.error(f"[{request_id}] Exception during direct variation request: {str(e)}")
                            # We return the error directly to the client instead of the transition to the backup path
                            return jsonify({
                                "error": f"Error processing direct variation request: {str(e)}"
                            }), 500
                    
                    # We convert the full URL to a relative path if it corresponds to the Asset.1Min.Ai format
                    image_path = None
                    if "asset.1min.ai" in image_url:
                        # We extract part of the path /images /...
                        path_match = re.search(r'(?:asset\.1min\.ai)(/images/[^?#]+)', image_url)
                        if path_match:
                            image_path = path_match.group(1)
                            # We remove the initial slash if it is
                            if image_path.startswith('/'):
                                image_path = image_path[1:]
                        else:
                            # We try to extract the path from the URL in general
                            path_match = re.search(r'/images/[^?#]+', image_url)
                            if path_match:
                                image_path = path_match.group(0)
                                # We remove the initial slash if it is
                                if image_path.startswith('/'):
                                    image_path = image_path[1:]

                    # If you find a relative path, we use it instead of a complete URL
                    download_url = image_url
                    if image_path:
                        logger.debug(f"[{request_id}] Extracted relative path from image URL: {image_path}")
                        # We use the full URL for loading, but we keep the relative path

                    # Download the image to a temporary file and send a redirection
                    # On the route/v1/images/variations by analogy s/v1/images/generations
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img_response = requests.get(download_url, stream=True)

                    if img_response.status_code != 200:
                        return jsonify(
                            {"error": f"Failed to download image from URL. Status code: {img_response.status_code}"}), 400

                    with open(temp_file.name, 'wb') as f:
                        for chunk in img_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # We save the path to the temporary file in memory for use in the route/v1/images/variations
                    variation_key = f"variation:{request_id}"
                    variation_data = {
                        "temp_file": temp_file.name,
                        "model": model,
                        "n": request_data.get("n", 1),
                        "image_path": image_path  # We keep the relative path if it is
                    }
                    # We use Safe_MemCeched_Operation, which now supports Memory_storage
                    safe_memcached_operation('set', variation_key, variation_data, expiry=300)  # Store 5 minutes
                    logger.debug(f"[{request_id}] Saved variation data with key: {variation_key}")

                    # We redirect the route/v1/images/variations
                    logger.info(f"[{request_id}] Redirecting to /v1/images/variations with model {model}")
                    
                    # Add detailed logistics for diagnosis
                    logger.info(f"[{request_id}] Temp file path: {temp_file.name}, exists: {os.path.exists(temp_file.name)}")
                    logger.info(f"[{request_id}] Image path: {image_path}")
                    logger.info(f"[{request_id}] Variation data prepared with temp file and image path")
                    
                    return redirect(url_for('image_variations', request_id=request_id), code=307)

            except Exception as e:
                logger.error(f"[{request_id}] Error processing variation command: {str(e)}")
                return jsonify({"error": f"Failed to process variation command: {str(e)}"}), 500


        # We log in the extracted Prompt for debugging
        logger.debug(f"[{request_id}] Extracted prompt text: {prompt_text[:100]}..." if len(
            prompt_text) > 100 else f"[{request_id}] Extracted prompt text: {prompt_text}")

        # We check whether the model belongs to one of the special types
        # For images generation models
        if model in IMAGE_GENERATION_MODELS:
            logger.info(f"[{request_id}] Redirecting image generation model to /v1/images/generations")

            # We create a new request only with the necessary fields to generate image
            # We take only the current user's current production without combining with history
            image_request = {
                "model": model,
                "prompt": prompt_text,  # Only the current request
                "n": request_data.get("n", 1),
                "size": request_data.get("size", "1024x1024")
            }

            # Add additional parameters for certain models
            if model == "dall-e-3":
                image_request["quality"] = request_data.get("quality", "standard")
                image_request["style"] = request_data.get("style", "vivid")

            # We check the availability of special parameters in Prompt for models type Midjourney
            if model.startswith("midjourney"):
                # Add inspections and parameters for midjourney models
                if "--ar" in prompt_text or "\u2014ar" in prompt_text:
                    logger.debug(f"[{request_id}] Found aspect ratio parameter in prompt")
                elif request_data.get("aspect_ratio"):
                    image_request["aspect_ratio"] = request_data.get("aspect_ratio")

                if "--no" in prompt_text or "\u2014no" in prompt_text:
                    logger.debug(f"[{request_id}] Found negative prompt parameter in prompt")
                elif request_data.get("negative_prompt"):
                    # Add negative prompt field as a separate parameter
                    image_request["negative_prompt"] = request_data.get("negative_prompt")

            # We delete messages from the request to avoid combining history
            if "messages" in image_request:
                del image_request["messages"]

            logger.debug(f"[{request_id}] Final image request: {json.dumps(image_request)[:200]}...")

            # We save a modified request (only the last request without history)
            request.environ["body_copy"] = json.dumps(image_request)
            return redirect(url_for('generate_image'), code=307)  # 307 preserves the method and body of the request

        # For speech generation models (TTS)
        if model in TEXT_TO_SPEECH_MODELS:
            logger.info(f"[{request_id}] Processing text-to-speech request directly")
            
            if not prompt_text:
                logger.error(f"[{request_id}] No input text provided for TTS")
                return jsonify({"error": "No input text provided"}), 400
                
            logger.debug(f"[{request_id}] TTS input text: {prompt_text[:100]}..." if len(prompt_text) > 100 else f"[{request_id}] TTS input text: {prompt_text}")
            
            voice = request_data.get("voice", "alloy")
            response_format = request_data.get("response_format", "mp3")
            speed = request_data.get("speed", 1.0)
            
            # We form Payload for a request to the API 1min.ai according to the documentation
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
                # Send the request directly
                logger.debug(f"[{request_id}] Sending direct TTS request to {ONE_MIN_API_URL}")
                response = api_request("POST", ONE_MIN_API_URL, json=payload, headers=headers)
                logger.debug(f"[{request_id}] TTS response status code: {response.status_code}")
                
                if response.status_code != 200:
                    if response.status_code == 401:
                        return ERROR_HANDLER(1020, key=api_key)
                    logger.error(f"[{request_id}] Error in TTS response: {response.text[:200]}")
                    return (
                        jsonify({"error": response.json().get("error", "Unknown error")}),
                        response.status_code,
                    )
                
                # We get a URL audio from the answer
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
                
                # Instead of downloading audio, we form a response with Markdown
                logger.info(f"[{request_id}] Successfully generated speech audio URL: {audio_url}")
                
                # We get a full URL for the audio file
                try:
                    # We check for the presence of a complete signed link in the response of the API
                    signed_url = None
                    
                    # Check the availability of the Temporaryurl field in the answer (according to the API response format)
                    if "temporaryUrl" in one_min_response:
                        signed_url = one_min_response["temporaryUrl"]
                        logger.debug(f"[{request_id}] Found temporaryUrl in API response root")
                    elif "result" in one_min_response and "resultList" in one_min_response["result"]:
                        # Check in the list of results
                        for item in one_min_response["result"]["resultList"]:
                            if item.get("type") == "TEXT_TO_SPEECH" and "temporaryUrl" in item:
                                signed_url = item["temporaryUrl"]
                                logger.debug(f"[{request_id}] Found temporaryUrl in resultList")
                                break
                    
                    # Checking in Airecord, if there are no links in the main places
                    if not signed_url and "aiRecord" in one_min_response:
                        if "temporaryUrl" in one_min_response["aiRecord"]:
                            signed_url = one_min_response["aiRecord"]["temporaryUrl"]
                            logger.debug(f"[{request_id}] Found temporaryUrl in aiRecord")
                    
                    # We check other possible fields for reverse compatibility
                    if not signed_url:
                        # We are looking for in various places in the API response format
                        if "aiRecord" in one_min_response and "aiRecordDetail" in one_min_response["aiRecord"]:
                            if "signedUrls" in one_min_response["aiRecord"]["aiRecordDetail"]:
                                signed_urls = one_min_response["aiRecord"]["aiRecordDetail"]["signedUrls"]
                                if isinstance(signed_urls, list) and signed_urls:
                                    signed_url = signed_urls[0]
                                elif isinstance(signed_urls, str):
                                    signed_url = signed_urls
                            elif "signedUrl" in one_min_response["aiRecord"]["aiRecordDetail"]:
                                signed_url = one_min_response["aiRecord"]["aiRecordDetail"]["signedUrl"]
                        elif "signedUrls" in one_min_response:
                            signed_urls = one_min_response["signedUrls"]
                            if isinstance(signed_urls, list) and signed_urls:
                                signed_url = signed_urls[0]
                            elif isinstance(signed_urls, str):
                                signed_url = signed_urls
                        elif "signedUrl" in one_min_response:
                            signed_url = one_min_response["signedUrl"]
                    
                    # We use the received signed link or basic URL
                    if signed_url:
                        full_audio_url = signed_url
                        logger.debug(f"[{request_id}] Using signed URL from API: {signed_url[:100]}...")
                    else:
                        # If there is no signed link, we use the basic URL in S3 format
                        # Although without a signature, he will most likely not work
                        full_audio_url = f"https://s3.us-east-1.amazonaws.com/asset.1min.ai/{audio_url}"
                        logger.warning(f"[{request_id}] No signed URL found, using base S3 URL: {full_audio_url}")
                
                except Exception as e:
                    logger.error(f"[{request_id}] Error processing audio URL: {str(e)}")
                    full_audio_url = f"https://asset.1min.ai/{audio_url}"
                    logger.warning(f"[{request_id}] Error occurred, using fallback URL: {full_audio_url}")
                
                # We form a response in the format similar to Chat Complets
                completion_response = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant", 
                                "content": f"🔊 [Audio.mp3]({full_audio_url})"
                            },
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

        # For models of audio transcription (STT)
        if model in SPEECH_TO_TEXT_MODELS:
            logger.info(f"[{request_id}] Redirecting speech-to-text model to /v1/audio/transcriptions")
            return redirect(url_for('audio_transcriptions'), code=307)

        # Let's journal the beginning of the request
        logger.debug(f"[{request_id}] Processing chat completion request")

        # Check whether the image of the image contains
        image = False
        image_paths = []

        # Check the availability of user files for working with PDF
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
                            # We extract the ID files
                            user_file_ids = [file_info.get("id") for file_info in user_files if file_info.get("id")]
                            logger.debug(f"[{request_id}] Found user files: {user_file_ids}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error parsing user files from memcached: {str(e)}")
            except Exception as e:
                logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")
        else:
            logger.debug(f"[{request_id}] Memcached not available, no user files loaded")

        # We check the availability of messages before the start of processing
        if not messages:
            logger.error(f"[{request_id}] No messages provided in request")
            return ERROR_HANDLER(1412)

        # We extract the text of the request for analysis
        extracted_prompt = messages[-1].get("content", "")
        if isinstance(extracted_prompt, list):
            extracted_prompt = " ".join([item.get("text", "") for item in extracted_prompt if "text" in item])
        extracted_prompt_lower = extracted_prompt.lower() if extracted_prompt else ""

        # If the request does not indicate File_ids, but the user has uploaded files,
        # Add them to the request only if the message mentions something about files or documents
        file_keywords = ["файл", "файлы", "file", "files", "документ", "документы", "document", "documents"]
        prompt_has_file_keywords = False

        # Check the availability of keywords about files in the request
        if extracted_prompt_lower:
            prompt_has_file_keywords = any(keyword in extracted_prompt_lower for keyword in file_keywords)

        # Add files only if the user requested work with files or clearly indicated File_ids
        if (not request_data.get("file_ids") and user_file_ids and prompt_has_file_keywords):
            logger.info(f"[{request_id}] Adding user files to request: {user_file_ids}")
            request_data["file_ids"] = user_file_ids
        elif not request_data.get("file_ids") and user_file_ids:
            logger.debug(f"[{request_id}] User has files but didn't request to use them in this message")

        # We get the contents of the last message for further processing
        user_input = messages[-1].get("content")
        if not user_input:
            logger.error(f"[{request_id}] No content in last message")
            return ERROR_HANDLER(1423)

        # We form the history of dialogue
        all_messages = format_conversation_history(
            request_data.get("messages", []), request_data.get("new_input", "")
        )

        # Checking for the presence of images in the last message
        if isinstance(user_input, list):
            logger.debug(
                f"[{request_id}] Processing message with multiple content items (text/images)"
            )
            combined_text = ""
            for i, item in enumerate(user_input):
                if "text" in item:
                    combined_text += item["text"] + "\n"
                    logger.debug(f"[{request_id}] Added text content from item {i + 1}")

                if "image_url" in item:
                    if model not in vision_supported_models:
                        logger.error(
                            f"[{request_id}] Model {model} does not support images"
                        )
                        return ERROR_HANDLER(1044, model)

                    # Create a hash url image for caching
                    image_key = None
                    image_url = None

                    # We extract the URL images
                    if (
                            isinstance(item["image_url"], dict)
                            and "url" in item["image_url"]
                    ):
                        image_url = item["image_url"]["url"]
                    else:
                        image_url = item["image_url"]

                    # Heshchit url for the cache
                    if image_url:
                        image_key = hashlib.md5(image_url.encode("utf-8")).hexdigest()

                    # Check the cache
                    if image_key and image_key in IMAGE_CACHE:
                        cached_path = IMAGE_CACHE[image_key]
                        logger.debug(
                            f"[{request_id}] Using cached image path for item {i + 1}: {cached_path}"
                        )
                        image_paths.append(cached_path)
                        image = True
                        continue

                    # We load the image if it is not in the cache
                    logger.debug(
                        f"[{request_id}] Processing image URL in item {i + 1}: {image_url[:30]}..."
                    )

                    # We load the image
                    image_path = retry_image_upload(
                        image_url, api_key, request_id=request_id
                    )

                    if image_path:
                        # We save in the cache
                        if image_key:
                            IMAGE_CACHE[image_key] = image_path
                            # Clean the old notes if necessary
                            if len(IMAGE_CACHE) > MAX_CACHE_SIZE:
                                old_key = next(iter(IMAGE_CACHE))
                                del IMAGE_CACHE[old_key]

                        image_paths.append(image_path)
                        image = True
                        logger.debug(
                            f"[{request_id}] Image {i + 1} successfully processed: {image_path}"
                        )
                    else:
                        logger.error(f"[{request_id}] Failed to upload image {i + 1}")

            # We replace user_input with the textual part only if it is not empty
            if combined_text:
                user_input = combined_text

        # We check if there is File_ids for a chat with documents
        file_ids = request_data.get("file_ids", [])
        conversation_id = request_data.get("conversation_id", None)

        # We extract the text of the request for the analysis of keywords
        prompt_text = all_messages.lower()
        extracted_prompt = messages[-1].get("content", "")
        if isinstance(extracted_prompt, list):
            extracted_prompt = " ".join([item.get("text", "") for item in extracted_prompt if "text" in item])
        extracted_prompt = extracted_prompt.lower()

        logger.debug(f"[{request_id}] Extracted prompt text: {extracted_prompt}")

        # We check the file deletion request
        delete_keywords = ["удалить", "удали", "удаление", "очисти", "очистка", "delete", "remove", "clean"]
        file_keywords = ["файл", "файлы", "file", "files", "документ", "документы", "document", "documents"]
        mime_type_keywords = ["pdf", "txt", "doc", "docx", "csv", "xls", "xlsx", "json", "md", "html", "htm", "xml",
                              "pptx", "ppt", "rtf"]

        # Combine all keywords for files
        all_file_keywords = file_keywords + mime_type_keywords

        # We check the request for file deletion (there must be keywords of deletion and file keywords)
        has_delete_keywords = any(keyword in extracted_prompt for keyword in delete_keywords)
        has_file_keywords = any(keyword in extracted_prompt for keyword in all_file_keywords)

        if has_delete_keywords and has_file_keywords and user_file_ids:
            logger.info(f"[{request_id}] Deletion request detected, removing all user files")

            # Trying to get ID teams
            team_id = None
            try:
                # Trying to get ID commands through API
                teams_url = f"{ONE_MIN_API_URL}/teams"
                teams_headers = {"API-KEY": api_key}
                teams_response = api_request("GET", teams_url, headers=teams_headers)
                if teams_response.status_code == 200:
                    teams_data = teams_response.json()
                    if "data" in teams_data and teams_data["data"]:
                        team_id = teams_data["data"][0].get("id")
                        logger.debug(f"[{request_id}] Found team ID for deletion: {team_id}")
            except Exception as e:
                logger.error(f"[{request_id}] Error getting team ID for deletion: {str(e)}")

            deleted_files = []
            for file_id in user_file_ids:
                try:
                    # We form a URL to delete the file depending on the availability of Team_id
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

            # Clean the user's list of user files in Memcache
            if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None and deleted_files:
                try:
                    user_key = f"user:{api_key}"
                    safe_memcached_operation('set', user_key, json.dumps([]))
                    logger.info(f"[{request_id}] Cleared user files list in memcached")
                except Exception as e:
                    logger.error(f"[{request_id}] Error clearing user files in memcached: {str(e)}")

            # Send a response to file deletion
            return jsonify({
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
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
                    "prompt_tokens": calculate_token(prompt_text),
                    "completion_tokens": 20,
                    "total_tokens": calculate_token(prompt_text) + 20
                }
            }), 200

        # We check the request for keywords for file processing
        has_file_reference = any(keyword in extracted_prompt for keyword in all_file_keywords)

        # If there is File_ids and the request contains keywords about files or there are ID conversations, we use Chat_with_PDF
        if file_ids and len(file_ids) > 0:
            logger.debug(
                f"[{request_id}] Creating CHAT_WITH_PDF request with {len(file_ids)} files"
            )

            # Add instructions for working with documents to Prompt
            enhanced_prompt = all_messages
            if not enhanced_prompt.strip().startswith(DOCUMENT_ANALYSIS_INSTRUCTION):
                enhanced_prompt = f"{DOCUMENT_ANALYSIS_INSTRUCTION}\n\n{all_messages}"

            # We get the user Team_id
            team_id = None
            try:
                teams_url = "https://api.1min.ai/api/teams"  # Correct URL C /API /
                teams_headers = {"API-KEY": api_key, "Content-Type": "application/json"}

                logger.debug(f"[{request_id}] Fetching team ID from: {teams_url}")
                teams_response = requests.get(teams_url, headers=teams_headers)

                if teams_response.status_code == 200:
                    teams_data = teams_response.json()
                    if "data" in teams_data and teams_data["data"]:
                        team_id = teams_data["data"][0].get("id")
                        logger.debug(f"[{request_id}] Got team ID: {team_id}")
                else:
                    logger.warning(
                        f"[{request_id}] Failed to get team ID: {teams_response.status_code} - {teams_response.text}")
            except Exception as e:
                logger.error(f"[{request_id}] Error getting team ID: {str(e)}")

            # If there is no Conversation_id, we create a new conversation
            if not conversation_id:
                conversation_id = create_conversation_with_files(
                    file_ids, "Chat with documents", model, api_key, request_id
                )
                if not conversation_id:
                    return (
                        jsonify({"error": "Failed to create conversation with files"}),
                        500,
                    )

            # We form Payload to request files
            payload = {"message": enhanced_prompt}
            if conversation_id:
                payload["conversationId"] = conversation_id

            # We use the correct URL API C /API /
            api_url = "https://api.1min.ai/api/features/conversations/messages"
            # Add Conversationid as a request parameter
            api_params = {"conversationId": conversation_id}

            logger.debug(
                f"[{request_id}] Sending message to conversation using URL: {api_url} with params: {api_params}")

            headers = {"API-KEY": api_key, "Content-Type": "application/json"}

            # Depending on the Stream parameter, select the request method
            if stream:
                # Streaming request
                return streaming_request(
                    api_url, payload, headers, request_id, model, model_settings, api_params=api_params
                )
            else:
                # The usual request
                try:
                    response = requests.post(api_url, json=payload, headers=headers, params=api_params)

                    logger.debug(f"[{request_id}] API response status code: {response.status_code}")
                    if response.status_code != 200:
                        logger.error(
                            f"[{request_id}] API error: {response.status_code} - {response.text}"
                        )
                        return (
                            jsonify({"error": "API request failed", "details": response.text}),
                            response.status_code,
                        )

                    # We convert the answer to the Openai format
                    response_data = response.json()
                    logger.debug(f"[{request_id}] Raw API response: {json.dumps(response_data)[:500]}...")

                    # We extract a response from different places of data structure
                    ai_response = None
                    if "answer" in response_data:
                        ai_response = response_data["answer"]
                    elif "message" in response_data:
                        ai_response = response_data["message"]
                    elif "result" in response_data:
                        ai_response = response_data["result"]
                    elif "aiRecord" in response_data and "aiRecordDetail" in response_data["aiRecord"]:
                        ai_response = response_data["aiRecord"]["aiRecordDetail"].get("answer", "")

                    if not ai_response:
                        # Recursively looking for a response on Keys Asswer, Message, Result
                        def find_response(obj, path=""):
                            if isinstance(obj, dict):
                                for key in ["answer", "message", "result"]:
                                    if key in obj:
                                        logger.debug(f"[{request_id}] Found response at path '{path}.{key}'")
                                        return obj[key]

                                for key, value in obj.items():
                                    result = find_response(value, f"{path}.{key}")
                                    if result:
                                        return result
                            elif isinstance(obj, list):
                                for i, item in enumerate(obj):
                                    result = find_response(item, f"{path}[{i}]")
                                    if result:
                                        return result
                            return None

                        ai_response = find_response(response_data)

                    if not ai_response:
                        logger.error(f"[{request_id}] Could not extract AI response from API response")
                        return jsonify({"error": "Could not extract AI response"}), 500

                    openai_response = format_openai_response(
                        ai_response, model, request_id
                    )
                    return jsonify(openai_response)
                except Exception as e:
                    logger.error(
                        f"[{request_id}] Exception while processing API response: {str(e)}"
                    )
                    traceback.print_exc()
                    return jsonify({"error": str(e)}), 500

        # Counting tokens
        prompt_token = calculate_token(str(all_messages))

        # Checking the model
        if PERMIT_MODELS_FROM_SUBSET_ONLY and model not in AVAILABLE_MODELS:
            return ERROR_HANDLER(1002, model)

        logger.debug(
            f"[{request_id}] Processing {prompt_token} prompt tokens with model {model}"
        )

        # Prepare Payload, taking into account the capabilities of the model
        payload = prepare_payload(
            request_data, model, all_messages, image_paths, request_id
        )

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}

        # Request depending on Stream
        if not request_data.get("stream", False):
            # The usual request
            logger.debug(
                f"[{request_id}] Sending non-streaming request to {ONE_MIN_API_URL}"
            )

            try:
                response = api_request(
                    "POST", ONE_MIN_API_URL, json=payload, headers=headers
                )
                logger.debug(
                    f"[{request_id}] Response status code: {response.status_code}"
                )

                if response.status_code != 200:
                    if response.status_code == 401:
                        return ERROR_HANDLER(1020, key=api_key)
                    try:
                        error_content = response.json()
                        logger.error(f"[{request_id}] Error response: {error_content}")
                    except:
                        logger.error(
                            f"[{request_id}] Could not parse error response as JSON"
                        )
                    return ERROR_HANDLER(response.status_code)

                one_min_response = response.json()
                transformed_response = transform_response(
                    one_min_response, request_data, prompt_token
                )

                response = make_response(jsonify(transformed_response))
                set_response_headers(response)
                return response, 200
            except Exception as e:
                logger.error(f"[{request_id}] Exception during request: {str(e)}")
                return jsonify({"error": str(e)}), 500
        else:
            # Streaming request
            logger.debug(f"[{request_id}] Sending streaming request")

            # URL for streaming mode
            streaming_url = f"{ONE_MIN_API_URL}?isStreaming=true"

            logger.debug(f"[{request_id}] Streaming URL: {streaming_url}")
            logger.debug(f"[{request_id}] Payload: {json.dumps(payload)[:200]}...")

            # If a web pion is included, we display a full websearch block for debugging
            if "promptObject" in payload and payload["promptObject"].get("webSearch"):
                logger.info(f"[{request_id}] Web search parameters in payload: " +
                            f"webSearch={payload['promptObject'].get('webSearch')}, " +
                            f"numOfSite={payload['promptObject'].get('numOfSite')}, " +
                            f"maxWord={payload['promptObject'].get('maxWord')}")

            try:
                # We use a session to control the connection
                session = create_session()
                response_stream = session.post(
                    streaming_url, json=payload, headers=headers, stream=True
                )

                logger.debug(
                    f"[{request_id}] Streaming response status code: {response_stream.status_code}"
                )

                if response_stream.status_code != 200:
                    if response_stream.status_code == 401:
                        session.close()
                        return ERROR_HANDLER(1020, key=api_key)

                    logger.error(
                        f"[{request_id}] Error status code: {response_stream.status_code}"
                    )
                    try:
                        error_content = response_stream.json()
                        logger.error(f"[{request_id}] Error response: {error_content}")
                    except:
                        logger.error(
                            f"[{request_id}] Could not parse error response as JSON"
                        )

                    session.close()
                    return ERROR_HANDLER(response_stream.status_code)

                # We transfer the session to Generator
                return Response(
                    stream_response(
                        response_stream, request_data, model, prompt_token, session
                    ),
                    content_type="text/event-stream",
                )
            except Exception as e:
                logger.error(
                    f"[{request_id}] Exception during streaming request: {str(e)}"
                )
                return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(
            f"[{request_id}] Exception during conversation processing: {str(e)}"
        )
        traceback.print_exc()
        return (
            jsonify({"error": f"Error during conversation processing: {str(e)}"}),
            500,
        )

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

    # Creating a conversation with PDF in 1min.ai
    payload = {
        "title": name,
        "type": "CHAT_WITH_PDF",
        "model": model,
        "fileList": file_ids,
    }

    response = requests.post(
        ONE_MIN_CONVERSATION_API_URL, json=payload, headers=headers
    )

    if response.status_code != 200:
        if response.status_code == 401:
            return ERROR_HANDLER(1020, key=api_key)
        return (
            jsonify({"error": response.json().get("error", "Unknown error")}),
            response.status_code,
        )

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

        response = make_response(jsonify(openai_response))
        set_response_headers(response)
        return response, 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Вспомогательные функции для текстовых моделей
def format_conversation_history(messages, new_input):
    """
    Formats the conversation history into a structured string.

    Args:
        messages (list): List of message dictionaries from the request
        new_input (str): The new user input message

    Returns:
        str: Formatted conversation history
    """
    formatted_history = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        # Handle potential list content
        if isinstance(content, list):
            processed_content = []
            for item in content:
                if "text" in item:
                    processed_content.append(item["text"])
            content = "\n".join(processed_content)

        if role == "system":
            formatted_history.append(f"System: {content}")
        elif role == "user":
            formatted_history.append(f"User: {content}")
        elif role == "assistant":
            formatted_history.append(f"Assistant: {content}")

    # Add new input if it is
    if new_input:
        formatted_history.append(f"User: {new_input}")

    # We return only the history of dialogue without additional instructions
    return "\n".join(formatted_history)


def get_model_capabilities(model):
    """
    Defines supported opportunities for a specific model

    Args:
        Model: The name of the model

    Returns:
        DICT: Dictionary with flags of supporting different features
    """
    capabilities = {
        "vision": False,
        "code_interpreter": False,
        "retrieval": False,
        "function_calling": False,
    }

    # We check the support of each opportunity through the corresponding arrays
    capabilities["vision"] = model in vision_supported_models
    capabilities["code_interpreter"] = model in code_interpreter_supported_models
    capabilities["retrieval"] = model in retrieval_supported_models
    capabilities["function_calling"] = model in function_calling_supported_models

    return capabilities


def prepare_payload(
        request_data, model, all_messages, image_paths=None, request_id=None
):
    """
    Prepares Payload for request, taking into account the capabilities of the model

    Args:
        Request_Data: Request data
        Model: Model
        All_Messages: Posts of Posts
        image_paths: ways to images
        Request_id: ID query

    Returns:
        DICT: Prepared Payload
    """
    capabilities = get_model_capabilities(model)

    # Check the availability of Openai tools
    tools = request_data.get("tools", [])
    web_search = False
    code_interpreter = False

    if tools:
        for tool in tools:
            tool_type = tool.get("type", "")
            # Trying to include functions, but if they are not supported, we just log in
            if tool_type == "retrieval":
                if capabilities["retrieval"]:
                    web_search = True
                    logger.debug(
                        f"[{request_id}] Enabled web search due to retrieval tool"
                    )
                else:
                    logger.debug(
                        f"[{request_id}] Model {model} does not support web search, ignoring retrieval tool"
                    )
            elif tool_type == "code_interpreter":
                if capabilities["code_interpreter"]:
                    code_interpreter = True
                    logger.debug(f"[{request_id}] Enabled code interpreter")
                else:
                    logger.debug(
                        f"[{request_id}] Model {model} does not support code interpreter, ignoring tool"
                    )
            else:
                logger.debug(f"[{request_id}] Ignoring unsupported tool: {tool_type}")

    # We check the direct parameters 1min.ai
    if not web_search and request_data.get("web_search", False):
        if capabilities["retrieval"]:
            web_search = True
        else:
            logger.debug(
                f"[{request_id}] Model {model} does not support web search, ignoring web_search parameter"
            )

    num_of_site = request_data.get("num_of_site", 3)
    max_word = request_data.get("max_word", 500)

    # We form the basic Payload
    if image_paths:
        # Even if the model does not support images, we try to send as a text request
        if capabilities["vision"]:
            # Add instructions to the prompt field
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

            if web_search:
                logger.debug(
                    f"[{request_id}] Web search enabled in payload with numOfSite={num_of_site}, maxWord={max_word}")
        else:
            logger.debug(
                f"[{request_id}] Model {model} does not support vision, falling back to text-only chat"
            )
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
                logger.debug(
                    f"[{request_id}] Web search enabled in payload with numOfSite={num_of_site}, maxWord={max_word}")
    elif code_interpreter:
        # If Code_interpreter is requested and supported
        payload = {
            "type": "CODE_GENERATOR",
            "model": model,
            "conversationId": "CODE_GENERATOR",
            "promptObject": {"prompt": all_messages},
        }
    else:
        # Basic text request
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
            logger.debug(
                f"[{request_id}] Web search enabled in payload with numOfSite={num_of_site}, maxWord={max_word}")

    return payload


def transform_response(one_min_response, request_data, prompt_token):
    try:
        # Output of the response structure for debugging
        logger.debug(f"Response structure: {json.dumps(one_min_response)[:200]}...")

        # We get an answer from the appropriate place to json
        result_text = (
            one_min_response.get("aiRecord", {})
            .get("aiRecordDetail", {})
            .get("resultObject", [""])[0]
        )

        if not result_text:
            # Alternative ways to extract an answer
            if "resultObject" in one_min_response:
                result_text = (
                    one_min_response["resultObject"][0]
                    if isinstance(one_min_response["resultObject"], list)
                    else one_min_response["resultObject"]
                )
            elif "result" in one_min_response:
                result_text = one_min_response["result"]
            else:
                # If you have not found an answer along the well -known paths, we return the error
                logger.error(f"Cannot extract response text from API result")
                result_text = "Error: Could not extract response from API"

        completion_token = calculate_token(result_text)
        logger.debug(
            f"Finished processing Non-Streaming response. Completion tokens: {str(completion_token)}"
        )
        logger.debug(f"Total tokens: {str(completion_token + prompt_token)}")

        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "mistral-nemo").strip(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result_text,
                    },
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
        # Return the error in the format compatible with Openai
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model", "mistral-nemo").strip(),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Error processing response: {str(e)}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_token,
                "completion_tokens": 0,
                "total_tokens": prompt_token,
            },
        }


def stream_response(response, request_data, model, prompt_tokens, session=None):
    """
    Stream received from 1min.ai response in a format compatible with Openai API.
    """
    all_chunks = ""

    # We send the first fragment: the role of the message
    first_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }

    yield f"data: {json.dumps(first_chunk)}\n\n"

    # Simple implementation for content processing
    for chunk in response.iter_content(chunk_size=1024):
        finish_reason = None

        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk.decode('utf-8')
                    },
                    "finish_reason": finish_reason
                }
            ]
        }
        all_chunks += chunk.decode('utf-8')
        yield f"data: {json.dumps(return_chunk)}\n\n"

    tokens = calculate_token(all_chunks)
    logger.debug(f"Finished processing streaming response. Completion tokens: {str(tokens)}")
    logger.debug(f"Total tokens: {str(tokens + prompt_tokens)}")

    # Final cup denoting the end of the flow
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": ""
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens + prompt_tokens
        }
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

def emulate_stream_response(full_content, request_data, model, prompt_tokens):
    """
    Emulates a streaming response for cases when the API does not support the flow gear

    Args:
        Full_Content: Full text of the answer
        Request_Data: Request data
        Model: Model
        Prompt_tokens: the number of tokens in the request

    Yields:
        STR: Lines for streaming
    """
    # We break the answer to fragments by ~ 5 words
    words = full_content.split()
    chunks = [" ".join(words[i: i + 5]) for i in range(0, len(words), 5)]

    for chunk in chunks:
        return_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": chunk}, "finish_reason": None}
            ],
        }

        yield f"data: {json.dumps(return_chunk)}\n\n"
        time.sleep(0.05)  # Small delay in emulating stream

    # We calculate the tokens
    tokens = calculate_token(full_content)

    # Final chambers
    final_chunk = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens,
            "total_tokens": tokens + prompt_tokens,
        },
    }

    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
