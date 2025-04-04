from flask import Blueprint

# Создание blueprints для разных разделов API
text_bp = Blueprint('text', __name__)
images_bp = Blueprint('images', __name__)
audio_bp = Blueprint('audio', __name__)
files_bp = Blueprint('files', __name__)

# Импорт функций из модулей маршрутов
# Note: These imports are here to make them available when importing from routes
# but are not used directly in __init__.py to avoid circular imports
from routes.text import (
    format_conversation_history, get_model_capabilities, prepare_payload,
    transform_response, stream_response, emulate_stream_response
)
from routes.images import parse_aspect_ratio, retry_image_upload, create_image_variations
from routes.files import upload_document, create_conversation_with_files

__all__ = [
    # Blueprints
    'text_bp', 'images_bp', 'audio_bp', 'files_bp',
    
    # Text module functions
    'format_conversation_history', 'get_model_capabilities', 'prepare_payload',
    'transform_response', 'stream_response', 'emulate_stream_response',
    
    # Images module functions
    'parse_aspect_ratio', 'retry_image_upload', 'create_image_variations',
    
    # Files module functions
    'upload_document', 'create_conversation_with_files'
]
