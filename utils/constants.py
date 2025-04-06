# utils/constants.py
# Файл констант для приложения

# Main URL for API
ONE_MIN_API_URL = "https://api.1min.ai/api/features"
ONE_MIN_ASSET_URL = "https://api.1min.ai/api/assets"
ONE_MIN_CONVERSATION_API_URL = "https://api.1min.ai/api/conversations"
ONE_MIN_CONVERSATION_API_STREAMING_URL = "https://api.1min.ai/api/features/stream"

# Add Constant Tamout used in the API_Request API
DEFAULT_TIMEOUT = 120 # 120 seconds for regular requests
MIDJOURNEY_TIMEOUT = 900  # 15 minutes for requests for Midjourney

# Constants for query types
IMAGE_GENERATOR = "IMAGE_GENERATOR"
IMAGE_VARIATOR = "IMAGE_VARIATOR"

# Максимальный размер кэша для изображений
MAX_CACHE_SIZE = 100

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

# Define the models that are available for use
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
    "whisper-1", # speech recognition
    "tts-1",     # Speech synthesis
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
    # "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",  # Replicate - Music Generation
    # "luma",                  # TTAPI - Luma
    # "Qubico/image-toolkit",  # TTAPI - Image Toolkit
    # "suno",                  # TTAPI - Suno Music
    # "kling",                 # TTAPI - Kling
    # "music-u",               # TTAPI - Music U
    # "music-s",               # TTAPI - Music S
    # "elevenlabs-tts"         # ElevenLabs - TTS
]

# Define the models that support vision inputs
vision_supported_models = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo"
]

# Define the models that support code interpreter
code_interpreter_supported_models = [
    "gpt-4o",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "deepseek-chat",
    "deepseek-reasoner"
]

# Define the models that support web search (retrieval)
retrieval_supported_models = [
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
function_calling_supported_models = [
    "gpt-4",
    "gpt-3.5-turbo"
]

# Determination of models for generating images
IMAGE_GENERATION_MODELS = [
    "dall-e-3",
    "dall-e-2",
    "stable-diffusion-xl-1024-v1-0",
    "stable-diffusion-v1-6",
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

# Models that support images
VARIATION_SUPPORTED_MODELS = [
    "midjourney",
    "midjourney_6_1",
    "dall-e-2",
    "clipdrop"
]

# We determine the Image_variation_Models Constant based on Variation_Supported_Models
IMAGE_VARIATION_MODELS = VARIATION_SUPPORTED_MODELS

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
    "tts-1"  # ,
    # "tts-1-hd",
    # "google-tts",
    # "elevenlabs-tts"
]

# Determination of models for speech recognition (STT)
SPEECH_TO_TEXT_MODELS = [
    "whisper-1"  # ,
    "latest_long",
    "latest_short",
    "phone_call",
    "telephony",
    "telephony_short",
    "medical_dictation",
    "medical_conversation"
]

# Default values
SUBSET_OF_ONE_MIN_PERMITTED_MODELS = ["mistral-nemo", "gpt-4o-mini", "o3-mini", "deepseek-chat"]
PERMIT_MODELS_FROM_SUBSET_ONLY = False
