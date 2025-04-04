# Маршруты для работы с файлами
@app.route("/v1/files", methods=["GET", "POST", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_files():
    # ...

@app.route("/v1/files/<file_id>", methods=["GET", "DELETE", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file(file_id):
    # ...

@app.route("/v1/files/<file_id>/content", methods=["GET", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file_content(file_id):
    # ...

@app.route("/v1/files", methods=["POST"])
@limiter.limit("60 per minute")
def upload_file():
    # ...

# Вспомогательные функции для работы с файлами
def upload_document(file_data, file_name, api_key, request_id=None):
    # ...

def create_conversation_with_files(file_ids, title, model, api_key, request_id=None):
    # ...
