from flask import Blueprint

# Создаем blueprints для разных разделов API
text_bp = Blueprint('text', __name__)
images_bp = Blueprint('images', __name__)
audio_bp = Blueprint('audio', __name__)
files_bp = Blueprint('files', __name__)

# Импортируем функции из соответствующих модулей
# Эти импорты также выполняют инициализацию маршрутов в blueprints
from routes.text import (
    index, models, create_chat_completion, create_assistant, format_conversation_history,
    get_model_capabilities, prepare_payload, transform_response, stream_response, 
    emulate_stream_response
)

from routes.images import (
    generate_image, image_variations, parse_aspect_ratio, retry_image_upload,
    create_image_variations
)

from routes.audio import (
    audio_transcriptions, audio_translations, text_to_speech
)

from routes.files import (
    handle_files, handle_file, handle_file_content, upload_file,
    upload_document, create_conversation_with_files
)

# Экспортируем все важные объекты
__all__ = [
    # Blueprints
    'text_bp', 'images_bp', 'audio_bp', 'files_bp',
    
    # Функции из text.py
    'index', 'models', 'create_chat_completion', 'create_assistant',
    'format_conversation_history', 'get_model_capabilities', 'prepare_payload',
    'transform_response', 'stream_response', 'emulate_stream_response',
    
    # Функции из images.py
    'generate_image', 'image_variations', 'parse_aspect_ratio',
    'retry_image_upload', 'create_image_variations',
    
    # Функции из audio.py
    'audio_transcriptions', 'audio_translations', 'text_to_speech',
    
    # Функции из files.py
    'handle_files', 'handle_file', 'handle_file_content', 'upload_file',
    'upload_document', 'create_conversation_with_files'
]
