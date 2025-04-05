# utils/common.py
# Общие утилиты
def calculate_token(sentence, model="DEFAULT"):
    """Calculate the number of tokens in a sentence based on the specified model."""

    if model.startswith("mistral"):
        # Initialize the Mistral tokenizer
        tokenizer = MistralTokenizer.v3(is_tekken=True)
        model_name = "open-mistral-nemo"  # Default to Mistral Nemo
        tokenizer = MistralTokenizer.from_model(model_name)
        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=[
                    UserMessage(content=sentence),
                ],
                model=model_name,
            )
        )
        tokens = tokenized.tokens
        return len(tokens)

    elif model in ["gpt-3.5-turbo", "gpt-4"]:
        # Use OpenAI's tiktoken for GPT models
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(sentence)
        return len(tokens)

    else:
        # Default to openai
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(sentence)
        return len(tokens)

# A function for performing a request to the API with a new session
def api_request(req_method, url, headers=None,
                requester_ip=None, data=None,
                files=None, stream=False,
                timeout=None, json=None, **kwargs):
    """Performs the HTTP request to the API with the normalization of the URL and error processing"""
    req_url = url.strip()
    logger.debug(f"API request URL: {req_url}")

    # Request parameters
    req_params = {}
    if headers:
        req_params["headers"] = headers
    if data:
        req_params["data"] = data
    if files:
        req_params["files"] = files
    if stream:
        req_params["stream"] = stream
    if json:
        req_params["json"] = json

    # Add other parameters
    req_params.update(kwargs)

    # We check whether the request is an operation with images
    is_image_operation = False
    if json and isinstance(json, dict):
        operation_type = json.get("type", "")
        if operation_type in [IMAGE_GENERATOR, IMAGE_VARIATOR]:
            is_image_operation = True
            logger.debug(f"Detected image operation: {operation_type}, using extended timeout")

    # We use increased timaut for operations with images
    if is_image_operation:
        req_params["timeout"] = timeout or MIDJOURNEY_TIMEOUT
        logger.debug(f"Using extended timeout for image operation: {MIDJOURNEY_TIMEOUT}s")
    else:
        req_params["timeout"] = timeout or DEFAULT_TIMEOUT

    # We fulfill the request
    try:
        response = requests.request(req_method, req_url, **req_params)
        return response
    except Exception as e:
        logger.error(f"API request error: {str(e)}")
        raise

def set_response_headers(response):
    response.headers["Content-Type"] = "application/json"
    response.headers["Access-Control-Allow-Origin"] = "*"  # Corrected the hyphen in the title name
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    # Add more Cors headings
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, Accept"
    return response  # Return the answer for the chain


def create_session():
    """Creates a new session with optimal settings for APIs"""
    session = requests.Session()

    # Setting up repeated attempts for all requests
    retry_strategy = requests.packages.urllib3.util.retry.Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def safe_temp_file(prefix, request_id=None):
    """
    Safely creates a temporary file and guarantees its deletion after use

    Args:
        Prefix: Prefix for file name
        Request_id: ID Request for Logging

    Returns:
        STR: Way to the temporary file
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

    # Create a temporary directory if it is not
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Clean old files (over 1 hour)
    try:
        current_time = time.time()
        for old_file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, old_file)
            if os.path.isfile(file_path):
                # If the file is older than 1 hour - delete
                if current_time - os.path.getmtime(file_path) > 3600:
                    try:
                        os.remove(file_path)
                        logger.debug(
                            f"[{request_id}] Removed old temp file: {file_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[{request_id}] Failed to remove old temp file {file_path}: {str(e)}"
                        )
    except Exception as e:
        logger.warning(f"[{request_id}] Error while cleaning old temp files: {str(e)}")

    # Create a new temporary file
    temp_file_path = os.path.join(temp_dir, f"{prefix}_{request_id}_{random_string}")
    return temp_file_path

def ERROR_HANDLER(code, model=None, key=None):
    # Handle errors in OpenAI-Structued Error
    error_codes = {  # Internal Error Codes
        1002: {
            "message": f"The model {model} does not exist.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_found",
            "http_code": 400,
        },
        1020: {
            "message": f"Incorrect API key provided: {key}. You can find your API key at https://app.1min.ai/api.",
            "type": "authentication_error",
            "param": None,
            "code": "invalid_api_key",
            "http_code": 401,
        },
        1021: {
            "message": "Invalid Authentication",
            "type": "invalid_request_error",
            "param": None,
            "code": None,
            "http_code": 401,
        },
        1212: {
            "message": f"Incorrect Endpoint. Please use the /v1/chat/completions endpoint.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_supported",
            "http_code": 400,
        },
        1044: {
            "message": f"This model does not support image inputs.",
            "type": "invalid_request_error",
            "param": None,
            "code": "model_not_supported",
            "http_code": 400,
        },
        1412: {
            "message": f"No message provided.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "invalid_request_error",
            "http_code": 400,
        },
        1423: {
            "message": f"No content in last message.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": "invalid_request_error",
            "http_code": 400,
        },
    }
    error_data = {
        k: v
        for k, v in error_codes.get(
            code,
            {
                "message": "Unknown error",
                "type": "unknown_error",
                "param": None,
                "code": None,
            },
        ).items()
        if k != "http_code"
    }  # Remove http_code from the error data
    logger.error(
        f"An error has occurred while processing the user's request. Error code: {code}"
    )
    return jsonify({"error": error_data}), error_codes.get(code, {}).get(
        "http_code", 400
    )  # Return the error data without http_code inside the payload and get the http_code to return.

def handle_options_request():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response, 204

def split_text_for_streaming(text, chunk_size=6):
    """
    It breaks the text into small parts to emulate streaming output.

    Args:
        Text (str): text for breakdown
        chunk_size (int): the approximate size of the parts in words

    Returns:
        List: List of parts of the text
    """
    if not text:
        return [""]

    # We break the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # We are grouping sentences to champs
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words_in_sentence = len(sentence.split())

        # If the current cup is empty or the addition of a sentence does not exceed the limit of words
        if not current_chunk or current_word_count + words_in_sentence <= chunk_size:
            current_chunk.append(sentence)
            current_word_count += words_in_sentence
        else:
            # We form a cup and begin the new
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = words_in_sentence

    # Add the last cup if it is not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # If there is no Cankov (breakdown did not work), we return the entire text entirely
    if not chunks:
        return [text]

    return chunks
