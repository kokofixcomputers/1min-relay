# Маршруты для работы с изображениями
@app.route("/v1/images/generations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
def generate_image():
    # ...

@app.route("/v1/images/variations", methods=["POST", "OPTIONS"])
@limiter.limit("60 per minute")
@cross_origin()
def image_variations():
    # ...

# Вспомогательные функции для изображений
def parse_aspect_ratio(prompt, model, request_data, request_id=None):
    # ...

def retry_image_upload(image_url, api_key, request_id=None):
    # ...

def create_image_variations(image_url, user_model, n, aspect_width=None, aspect_height=None, mode=None, request_id=None):
    # ...
