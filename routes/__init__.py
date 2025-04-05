from flask import Blueprint

# Создаем blueprints для разных разделов API
text_bp = Blueprint('text', __name__)
images_bp = Blueprint('images', __name__)
audio_bp = Blueprint('audio', __name__)
files_bp = Blueprint('files', __name__)

# Импортируем маршруты
from routes.text import *
from routes.images import *
from routes.audio import *
from routes.files import *

__all__ = [
    # Blueprints
    'text_bp', 'images_bp', 'audio_bp', 'files_bp',
]
