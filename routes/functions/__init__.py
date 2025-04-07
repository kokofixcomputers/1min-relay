# routes/functions/__init__.py
# Инициализация субпакета функций

# Экспортируем общие функции
from .shared_func import (
    validate_auth,
    handle_api_error,
    format_openai_response,
    format_image_response,
    stream_response,
    get_full_url,
    extract_data_from_api_response,
    extract_text_from_response,
    extract_image_urls,
    extract_audio_url
)

# Экспортируем функции для текстовых моделей
from .txt_func import (
    format_conversation_history,
    get_model_capabilities,
    prepare_payload,
    transform_response,
    emulate_stream_response
)

# Экспортируем функции для изображений
from .img_func import (
    build_generation_payload,
    parse_aspect_ratio,
    create_image_variations,
    build_img2img_payload,
    process_image_tool_calls
)

# Экспортируем функции для аудио
from .audio_func import (
    prepare_models_list,
    prepare_whisper_payload,
    prepare_tts_payload
)

# Экспортируем функции для файлов
from .file_func import (
    upload_asset,
    get_mime_type,
    retry_image_upload,
    upload_audio_file
) 
