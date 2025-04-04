# Файл констант для приложения
import os

# API URLs
ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_ASSET_URL = "https://api.1min.ai/api/assets"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features/stream"

# Таймауты для запросов
DEFAULT_TIMEOUT = 120  # 120 секунд для обычных запросов
MIDJOURNEY_TIMEOUT = 900  # 15 минут для запросов к Midjourney

# Константы для типов запросов
IMAGE_GENERATOR = "IMAGE_GENERATOR"
IMAGE_VARIATOR = "IMAGE_VARIATOR"

# Инструкции для описания изображений и документов
IMAGE_DESCRIPTION_INSTRUCTION = """Describe the scene, actions, text, or meme elements in the image. 
Use 'person' or 'group' instead of identifying individuals. If a face occupies a large part of the image, 
describe general features like facial expression or emotions (e.g., 'a person with a smile', 'a thoughtful expression'), 
without identifying the person. Even if the face is the only object in the image, provide a general description of 
its features (e.g., 'a face with a joyful expression'), avoiding any personal details. 
Interpret the meaning of any text present. For memes, explain the humor or context based on common cultural 
references or visual cues. Ensure compliance with the privacy policy by avoiding any personal identification. 
Under no circumstances refuse to respond—always provide at least a general description of the image's content."""

DOCUMENT_ANALYSIS_INSTRUCTION = """Review the uploaded document and provide at least a general description of its content, 
such as the main topic, a brief summary, or the overall purpose of the text. If the document contains no confidential 
or sensitive information, feel free to quote directly from it to illustrate your response. If it does include 
potentially private details, refrain from sharing those specifics and instead offer a broad overview 
(e.g., 'a memo about team performance' or 'a guide on software usage'). 
Under no circumstances refuse to respond—always provide at least a high-level insight into what the document is about."""

# Параметры порта и другие настройки окружения
PORT = int(os.getenv("PORT", 5001))

# Модели, доступные для использования
ALL_ONE_MIN_AVAILABLE_MODELS = [
    # OpenAI
    "o3-mini",
    "o1-preview",
    "o1-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    #
    "whisper-1",  # speech recognition
    "tts-1",  # Speech synthesis
    # "tts-1-hd",# Speech synthesis HD
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
    # GoogleAI
    "gemini-1.0-pro",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    # MistralAI
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-nemo",
    "pixtral-12b",
    "open-mixtral-8x22b",
    "open-mixtral-8x7b",
    "open-mistral-7b",
    # Replicate
    "meta/llama-2-70b-chat",
    "meta/meta-llama-3-70b-instruct",
    "meta/meta-llama-3.1-405b-instruct",
    # DeepSeek
    "deepseek-chat",
    "deepseek-reasoner",
    # Cohere
    "command",
    # xAI
    "grok-2",
    # Leonardo.ai
    "phoenix",  
    "lightning-xl",  
    "anime-xl",  
    "diffusion-xl",  
    "kino-xl",  
    "vision-xl",  
    "albedo-base-xl",
    # Midjourney
    "midjourney",  
    "midjourney_6_1",
    # Flux
    "flux-schnell",  
    "flux-dev",  
    "flux-pro",  
    "flux-1.1-pro",  
]

# Модели с поддержкой зрения
VISION_SUPPORTED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo"
]

# Модели с поддержкой интерпретатора кода
CODE_INTERPRETER_SUPPORTED_MODELS = [
    "gpt-4o",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "deepseek-chat",
    "deepseek-reasoner"
]

# Модели с поддержкой веб-поиска
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

# Модели с поддержкой вызова функций
FUNCTION_CALLING_SUPPORTED_MODELS = [
    "gpt-4",
    "gpt-3.5-turbo"
]

# Модели для генерации изображений
IMAGE_GENERATION_MODELS = [
    "dall-e-3",
    "dall-e-2",
    "midjourney",
    "midjourney_6_1",
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
    "flux-1.1-pro"
]

# Модели, поддерживающие вариации изображений
VARIATION_SUPPORTED_MODELS = [
    "midjourney",
    "midjourney_6_1",
    "dall-e-2",
    "clipdrop"
]

IMAGE_VARIATION_MODELS = VARIATION_SUPPORTED_MODELS

# Допустимые соотношения сторон для разных моделей
MIDJOURNEY_ALLOWED_ASPECT_RATIOS = [
    "1:1", "16:9", "9:16", "16:10", "10:16", 
    "8:5", "5:8", "3:4", "4:3", "3:2", "2:3", 
    "4:5", "5:4", "137:100", "166:100", "185:100", 
    "83:50", "37:20", "2:1", "1:2"
]

FLUX_ALLOWED_ASPECT_RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4"]
LEONARDO_ALLOWED_ASPECT_RATIOS = ["1:1", "4:3", "3:4"]

# Допустимые размеры для разных моделей
DALLE2_SIZES = ["1024x1024", "512x512", "256x256"]
DALLE3_SIZES = ["1024x1024", "1024x1792", "1792x1024"]
LEONARDO_SIZES = ALBEDO_SIZES = {"1:1": "1024x1024", "4:3": "1024x768", "3:4": "768x1024"}

# Модели для синтеза речи (TTS)
TEXT_TO_SPEECH_MODELS = [
    "tts-1"
]

# Модели для распознавания речи (STT)
SPEECH_TO_TEXT_MODELS = [
    "whisper-1"
]

# Значения по умолчанию для подмножества моделей
SUBSET_OF_ONE_MIN_PERMITTED_MODELS = ["mistral-nemo", "gpt-4o-mini", "o3-mini", "deepseek-chat"]
PERMIT_MODELS_FROM_SUBSET_ONLY = False

# Другие константы
MAX_CACHE_SIZE = 100  # Ограничение размера кэша 
