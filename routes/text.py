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
