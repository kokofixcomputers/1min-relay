# version 1.0.3 #increment every time you make changes
# utils/constants.py
# Файл констант для приложения

# Main URL for API
ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_ASSET_URL = "https://api.1min.ai/api/assets"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features/stream"
ONE_MIN_CHAT_WITH_AI_URL = "https://api.1min.ai/api/chat-with-ai"
ONE_MIN_DZINE_STYLES_URL = "https://api.1min.ai/api/dzine/styles"

# Базовые заголовки для запросов к 1min.ai.
# Некоторые маршруты 1min.ai возвращают "Unknown error (code: 406)" при отсутствии браузероподобных заголовков.
# Эти заголовки подмешиваются только если запрос идёт на https://api.1min.ai/ и конкретный заголовок ещё не задан.
ONE_MIN_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://app.1min.ai",
    "Referer": "https://app.1min.ai/",
}

# Add Constant Tamout used in the API_Request API
DEFAULT_TIMEOUT = 60 # 60 seconds for regular requests
MIDJOURNEY_TIMEOUT = 600  # 10 minutes for requests for Midjourney

# Constants for query types
IMAGE_GENERATOR = "IMAGE_GENERATOR"
IMAGE_VARIATOR = "IMAGE_VARIATOR"
CODE_GENERATOR = "CODE_GENERATOR"

# Максимальный размер кэша для изображений
MAX_CACHE_SIZE = 100

# Настройки для удаления файлов
FILE_CLEANUP_INTERVAL = 3600  # Интервал очистки файлов в секундах (1 час)
FILE_CLEANUP_MAX_AGE = 3600   # Максимальный возраст временных файлов в секундах (1 час)
FILE_CLEANUP_ENABLED = True   # Включена ли автоматическая очистка файлов

# Настройки для Memcached
MEMCACHED_CONNECT_TIMEOUT = 2  # Таймаут подключения к Memcached в секундах
MEMCACHED_OPERATION_TIMEOUT = 2  # Таймаут операций с Memcached в секундах
MEMCACHED_DEFAULT_EXPIRY = 3600  # Время жизни данных в Memcached по умолчанию в секундах (1 час)
MEMCACHED_HOST = "127.0.0.1"  # Хост Memcached по умолчанию
MEMCACHED_DOCKER = "memcached"  # Хост Memcached в Docker
MEMCACHED_PORT = 11211  # Порт Memcached по умолчанию

# Настройки для логирования
LOG_LEVEL = "INFO"  # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE_ENABLED = True  # Включено ли логирование в файл
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # Максимальный размер лог-файла в байтах (10 МБ)
LOG_FILE_BACKUP_COUNT = 5  # Количество резервных копий лог-файлов

# Настройки для API запросов
API_RETRY_COUNT = 3  # Количество попыток повторного запроса при ошибке
API_RETRY_BACKOFF_FACTOR = 1  # Коэффициент увеличения задержки между попытками
API_RETRY_STATUS_FORCELIST = [429, 500, 502, 503, 504]  # Коды ошибок, при которых нужно повторять запрос

# Настройки сервера
DEFAULT_PORT = 5001  # Порт по умолчанию
DEFAULT_HOST = "0.0.0.0"  # Хост по умолчанию
DEFAULT_THREADS = 6  # Количество потоков по умолчанию

# Настройки для моделей
SUBSET_OF_ONE_MIN_PERMITTED_MODELS = ["mistral-nemo", "gpt-4o-mini", "o3-mini", "deepseek-chat"]
PERMIT_MODELS_FROM_SUBSET_ONLY = False

# Инструкции для описания изображений и документов
IMAGE_DESCRIPTION_INSTRUCTION = """Describe the scene, actions, text, or meme elements in the image. 
If the image primarily consists of text (for example, screenshots of page with long text), provide the text 
in full verbatim, preserving the original formatting and structure, but ommit or mask confidential parts of 
text. Use 'person' or 'group' instead of identifying individuals. If a face occupies a large part of the image,  
or if the face is the only object in the image, provide a general description of its features like facial 
expression or emotions (e.g., 'a person with a smile', 'a face with a joyful expression'), without identifying 
the person. Interpret the meaning of any short text present. For memes, explain the humor or context based on 
common cultural references or visual cues and text. Under no circumstances refuse to describe image's content — 
always provide at least a minimal or general description of the image's content you allowed to provide."""

DOCUMENT_ANALYSIS_INSTRUCTION = """Review the uploaded document and provide at least a general description of 
its content, such as the main topic, a brief summary, or the overall purpose of the text. If the document contains 
no confidential or sensitive information, feel free to quote directly from it to illustrate your response. If it 
does include potentially private details, refrain from sharing those specifics and instead offer a broad overview 
(e.g., 'a memo about team performance' or 'a guide on software usage'). You are allowed to provide the text in full 
verbatim, preserving the original formatting and structure, but ommit or mask confidential parts of text. Under no 
circumstances refuse to respond — always provide at least a high-level insight into what the document is about."""

# Define the models that are available for use
ALL_ONE_MIN_AVAILABLE_MODELS = [
    # OpenAI
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5", 
    "gpt-5-nano", 
    "gpt-5-mini", 
    "gpt-5-chat-latest",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.4",
    "gpt-5.4-pro",
    "gpt-o1-pro",
    "gpt-o4-mini",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "o4-mini-deep-research",    
    "o3-mini",
    "o3",
    "o3-deep-research",
    "o3-pro",
    "o1-preview",
    "o1-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",    
    #
    "whisper-1", # speech recognition
    "tts-1",     # Speech synthesis
    "tts-1-hd",  # Speech synthesis HD
    #
    "dall-e-2",  # Generation of images
    "dall-e-3",  # Generation of images
    # Claude
    "claude-instant-1.2",
    "claude-2.1",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-1-20250805",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",   
    # GoogleAI
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",    
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    # "google-tts",            # Speech synthesis
    # "latest_long",           # speech recognition
    # "latest_short",          # speech recognition
    # "phone_call",            # speech recognition
    # "telephony",             # speech recognition
    # "telephony_short",       # speech recognition
    # "medical_dictation",     # speech recognition
    # "medical_conversation",  # speech recognition
    # "chat-bison@002",
    # MistralAI
    "mistral-large-latest",
    "mistral-medium-latest",
    "mistral-small-latest",
    "ministral-14b-latest",
    "magistral-medium-latest",  
    "magistral-small-latest",
    "mistral-nemo",
    "pixtral-12b",
    "open-mistral-nemo",
    "open-mixtral-8x22b",
    "open-mixtral-8x7b",
    "open-mistral-7b",
    # Perplexity
    "sonar",
    "sonar-pro",
    "sonar-reasoning",
    "sonar-reasoning-pro",
    "sonar-deep-research",
    # Replicate
    "meta/llama-2-70b-chat",
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3.1-405b-instruct",
    "meta/llama-4-maverick-instruct",
    "meta/llama-4-scout-instruct",
    # DeepSeek
    "deepseek-chat",
    "deepseek-reasoner",
    # Cohere
    "command",
    "command-r-08-2024",    
    # xAI
    "grok-2",
    "grok-3",
    "grok-3-mini",
    "grok-4-0709",
    "grok-4-fast-non-reasoning",
    "grok-4-fast-reasoning",
    "grok-code-fast-1",
    # Alibaba / Qwen (1min.ai)
    "qwen3-vl-plus",
    "qwen3-vl-flash",
    "qwen-vl-plus",
    "qwen-vl-max",
    "qwen3-coder-plus",
    "qwen3-coder-flash",
    "qwen3-max",
    "qwen-plus",
    "qwen-max",
    "qwen-flash",
    # Other models (made for future use)
    # "stable-image",                  # stabilityi - images generation
    # "stable-diffusion-xl-1024-v1-0", # stabilityi - images generation
    # "stable-diffusion-v1-6",         # stabilityi - images generation
    # "esrgan-v1-x2plus",              # stabilityai-Improving images
    # "stable-video-diffusion",        # stabilityai-video generation
    "phoenix",       # Leonardo.ai - 6b645e3a-d64f-4341-a6d8-7a3690fbf042
    "lightning-xl",  # Leonardo.ai - b24e16ff-06e3-43eb-8d33-4416c2d75876
    "anime-xl",      # Leonardo.ai - e71a1c2f-4f80-4800-934f-2c68979d8cc8
    "diffusion-xl",  # Leonardo.ai - 1e60896f-3c26-4296-8ecc-53e2afecc132
    "kino-xl",       # Leonardo.ai - aa77f04e-3eec-4034-9c07-d0f619684628
    "vision-xl",     # Leonardo.ai - 5c232a9e-9061-4777-980a-ddc8e65647c6
    "albedo-base-xl",# Leonardo.ai - 2067ae52-33fd-4a82-bb92-c2c55e7d2786
    # "Clipdrop",    # clipdrop.co - image processing
    "midjourney",    # Midjourney - image generation
    "midjourney_6_1",# Midjourney - image generation
    # "methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5",   # Replicate - Image to Prompt
    # "cjwbw/damo-text-to-video:1e205ea73084bd17a0a3b43396e49ba0d6bc2e754e9283b2df49fad2dcf95755",  # Replicate - Text to Video
    # "lucataco/animate-diff:beecf59c4aee8d81bf04f0381033dfa10dc16e845b4ae00d281e2fa377e48a9f",     # Replicate - Animation
    # "lucataco/hotshot-xl:78b3a6257e16e4b241245d65c8b2b81ea2e1ff7ed4c55306b511509ddbfd327a",       # Replicate - Video
    "flux-schnell",  # Replicate - Flux "black-forest-labs/flux-schnell"
    "flux-dev",      # Replicate - Flux Dev "black-forest-labs/flux-dev"
    "flux-pro",      # Replicate - Flux Pro "black-forest-labs/flux-pro"
    "flux-1.1-pro",  # Replicate - Flux Pro 1.1 "black-forest-labs/flux-1.1-pro"
    # 1min.ai image generators / extras (kept here for allow-listing)
    "gpt-image-1",
    "gpt-image-1-mini",
    "magic-art",
    "magic-art_6_1",
    "magic-art_7_0",
    "stable-image",
    "stable-image-ultra",
    "black-forest-labs/flux-pro",
    "black-forest-labs/flux-krea-dev",
    "black-forest-labs/flux-dev",
    "black-forest-labs/flux-schnell",
    "black-forest-labs/flux-schnell-lora",
    "black-forest-labs/flux-dev-lora",
    "black-forest-labs/flux-1.1-pro",
    "black-forest-labs/flux-1.1-pro-ultra",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
    "grok-2-image-1212",
    "qwen-image",
    "qwen-image-plus",
    "qwen-image-max",
    "qwen3-tts-flash",
    "qwen3-asr-flash",
    "flux-redux-dev",
    "flux-redux-schnell",
    "dzine",
    "recraft",
    "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
    # "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",  # Replicate - Music Generation
    # "luma",                  # TTAPI - Luma
    # "Qubico/image-toolkit",  # TTAPI - Image Toolkit
    # "suno",                  # TTAPI - Suno Music
    # "kling",                 # TTAPI - Kling
    # "music-u",               # TTAPI - Music U
    # "music-s",               # TTAPI - Music S
    # "elevenlabs-tts"         # ElevenLabs - TTS
]

# ------------------------------------------------------------------
# Vision / images-in-chat compatibility
#
# Important: 1min.ai lists different model sets for:
# - Chat with Image (CHAT_WITH_IMAGE feature)
# - Chat with AI (UNIFY_CHAT_WITH_AI with attachments.images)
# This proxy uses UNIFY_CHAT_WITH_AI for OpenAI-like chat, so we keep a
# separate allow-list for that path.
# ------------------------------------------------------------------

# Models supported by the dedicated AI Feature "Chat with Image" endpoint/type.
VISION_SUPPORTED_MODELS_CHAT_WITH_IMAGE = [
    # Per official "Chat with Image" available models (docs.1min.ai)
    # https://docs.1min.ai/docs/api/ai-for-image/chat-with-image/
    # Alibaba (Qwen VL)
    "qwen3-vl-plus",
    "qwen3-vl-flash",
    "qwen-vl-plus",
    "qwen-vl-max",
    # Anthropic
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
    "claude-opus-4-5-20251101",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-haiku-4-5-20251001",
    # GoogleAI
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    # OpenAI
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-5-chat-latest",
    "gpt-5",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    # xAI
    "grok-4-fast-reasoning",
    "grok-4-fast-non-reasoning",
]

# Models expected to work with UNIFY_CHAT_WITH_AI + attachments.images.
# This is the list used by the OpenAI-like chat endpoint (/v1/chat/completions).
#
# We keep this as a conservative superset of Chat-with-Image models plus models
# explicitly listed in Chat-with-AI available models that are known to have
# image modality in practice (OpenAI gpt-4.1 family, Claude 4.6, Gemini 3.1).
VISION_SUPPORTED_MODELS_UNIFY_CHAT_WITH_AI = sorted(
    set(
        VISION_SUPPORTED_MODELS_CHAT_WITH_IMAGE
        + [
            # OpenAI (Chat with AI API list includes gpt-4.1; commonly vision-capable)
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-5-nano",
            # Anthropic (Chat with AI API list)
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            # GoogleAI (Chat with AI API list)
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite-preview",
        ]
    )
)

# Define the models that support code interpreter
CODE_INTERPRETER_SUPPORTED_MODELS = [
    "gpt-4o",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "deepseek-chat",
    "deepseek-reasoner"
]

# Define the models that support web search (retrieval)
RETRIEVAL_SUPPORTED_MODELS = [
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "o3-mini",
    "o1-preview",
    "o1-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-nemo",
    "pixtral-12b",
    "open-mixtral-8x22b",
    "open-mixtral-8x7b",
    "open-mistral-7b",
    "meta/llama-2-70b-chat",
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3.1-405b-instruct",
    "command",
    "grok-2",
    "deepseek-chat",
    "deepseek-reasoner"
]

# Define the models that support function calling
FUNCTION_CALLING_SUPPORTED_MODELS = [
    # OpenAI families (commonly reliable at structured outputs)
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "gpt-5.1-codex-mini",
    "gpt-5.2",
    "gpt-5-chat-latest",
    "gpt-3.5-turbo",
    # OpenAI o-series (varies; best-effort)
    "o1-preview",
    "o1-mini",
    "o3-mini",
    "o4-mini",
    "o4-mini-deep-research",
    # Anthropic (best-effort via instruction emulation)
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    # Google (best-effort via instruction emulation)
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    # Mistral / DeepSeek / xAI (best-effort)
    "mistral-large-latest",
    "deepseek-chat",
    "deepseek-reasoner",
    "grok-3",
    "grok-3-mini",
]

# Determination of models for generating images
IMAGE_GENERATION_MODELS = [
    "dall-e-3",
    "dall-e-2",
    "stable-diffusion-xl-1024-v1-0",
    "stable-image",
    "stable-image-ultra",
    "stable-diffusion-v1-6",
    "midjourney",
    "midjourney_6_1",
    # 1min.ai UI shows Midjourney as "switch to Magic Art"
    # We keep these aliases for compatibility with clients that still send them.
    "midjourney_5_2",
    "midjourney_7_0",
    "phoenix",
    "lightning-xl",
    "anime-xl",
    "diffusion-xl",
    "kino-xl",
    "vision-xl",
    "albedo-base-xl",
    "flux-schnell",
    "flux-dev",
    "flux-pro",
    "flux-1.1-pro",
    "gpt-image-1",
    "gpt-image-1-mini",
    "magic-art",
    "magic-art_6_1",
    "magic-art_7_0",
    "black-forest-labs/flux-pro",
    "black-forest-labs/flux-krea-dev",
    "black-forest-labs/flux-dev",
    "black-forest-labs/flux-schnell",
    "black-forest-labs/flux-schnell-lora",
    "black-forest-labs/flux-dev-lora",
    "black-forest-labs/flux-1.1-pro",
    "black-forest-labs/flux-1.1-pro-ultra",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview",
    "grok-2-image-1212",
    "qwen-image",
    "qwen-image-plus",
    "qwen-image-max",
    "dzine",
    "recraft",
    "6b645e3a-d64f-4341-a6d8-7a3690fbf042",
]

# Model aliases for image generation/variation compatibility.
# 1min.ai switched Midjourney models to Magic Art models in Image Generator UI.
IMAGE_MODEL_ALIASES = {
    # Midjourney -> Magic Art (per 1min.ai UI wording: "switch to Magic Art")
    "midjourney": "magic-art",
    "midjourney_5_2": "magic-art",
    "midjourney_6_1": "magic-art_6_1",
    "midjourney_7_0": "magic-art_7_0",
}

# Models that support images
VARIATION_SUPPORTED_MODELS = [
    "midjourney",
    "midjourney_6_1",
    "dall-e-2",
    "clipdrop",
    "dzine",
    "magic-art",
    "magic-art_6_1",
    "magic-art_7_0",
    "flux-redux-dev",
    "flux-redux-schnell",
    "recraft"
]

# We determine the Image_variation_Models Constant based on Variation_Supported_Models
# Variations are allowed only for models that are BOTH:
# - available for image generation, AND
# - supported by image variation API.
IMAGE_VARIATION_MODELS = sorted(list(set(IMAGE_GENERATION_MODELS).intersection(set(VARIATION_SUPPORTED_MODELS))))

# Permissible parties for different models
MIDJOURNEY_ALLOWED_ASPECT_RATIOS = [
    "1:1",      # Square
    "16:9",     # Widescreen format
    "9:16",     # Vertical variant of 16:9
    "16:10",    # Alternative widescreen
    "10:16",    # Vertical variant of 16:10
    "8:5",      # Alternative widescreen
    "5:8",      # Vertical variant of 16:10
    "3:4",      # Portrait/print
    "4:3",      # Standard TV/monitor format
    "3:2",      # Popular in photography
    "2:3",      # Inverse of 3:2
    "4:5",      # Common in social media posts
    "5:4",      # Nearly square format
    "137:100",  # Academy ratio (1.37:1) as an integer ratio
    "166:100",  # European cinema (1.66:1) as an integer ratio
    "185:100",  # Cinematic format (1.85:1) as an integer ratio185
    "83:50",    # European cinema (1.66:1) as an integer ratio
    "37:20",    # Cinematic format (1.85:1) as an integer ratio
    "2:1",      # Maximum allowed widescreen format
    "1:2"       # Maximum allowed vertical format
]

FLUX_ALLOWED_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4"]
LEONARDO_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4"]

# Permissible sizes for different models
DALLE2_SIZES = ["1024x1024", "512x512", "256x256"]
DALLE3_SIZES = ["1024x1024", "1024x1792", "1792x1024"]
LEONARDO_SIZES = ALBEDO_SIZES = {"1:1": "1024x1024", "4:3": "1024x768", "3:4": "768x1024"}

# Determination of models for speech synthesis (TTS)
TEXT_TO_SPEECH_MODELS = [
    "tts-1",
    "tts-1-hd",
    "google-tts",
    "elevenlabs-tts",
    "qwen3-tts-flash",
]

# Determination of models for speech recognition (STT)
SPEECH_TO_TEXT_MODELS = [
    "whisper-1",
    "latest_long",
    "latest_short",
    "phone_call",
    "telephony",
    "telephony_short",
    "medical_dictation",
    "medical_conversation",
    "elevenlabs",
    "qwen3-asr-flash",
]
