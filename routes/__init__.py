from flask import Blueprint

# Создание blueprints для разных разделов API
text_bp = Blueprint('text', __name__)
images_bp = Blueprint('images', __name__)
audio_bp = Blueprint('audio', __name__)
files_bp = Blueprint('files', __name__)

# Импорт функций из модулей маршрутов
from routes.text import (
    format_conversation_history, get_model_capabilities, prepare_payload,
    transform_response, stream_response, emulate_stream_response
)
from routes.images import parse_aspect_ratio, retry_image_upload, create_image_variations
from routes.files import upload_document, create_conversation_with_files

__all__ = [
    'text_bp', 'images_bp', 'audio_bp', 'files_bp',
    'format_conversation_history', 'get_model_capabilities', 'prepare_payload',
    'transform_response', 'stream_response', 'emulate_stream_response',
    'parse_aspect_ratio', 'retry_image_upload', 'create_image_variations',
    'upload_document', 'create_conversation_with_files'
]
