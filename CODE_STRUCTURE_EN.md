# Project Code Structure

## Main application file: `1min-relay/app.py`
- Initializes the Flask app and starts the server
- Imports all the necessary modules
- Configures server settings

## Utilities

### Common Utilities: `1min-relay/utils/common.py`
- `ERROR_HANDLER`: Creates standardized error responses with appropriate status codes
- `handle_options_request`: Handles OPTIONS requests for CORS preflight
- `set_response_headers`: Sets response headers for CORS
- `create_session`: Creates a session for API requests
- `api_request`: Makes requests to external APIs with error handling
- `safe_temp_file`: Creates a temporary file with proper resource management
- `calculate_token`: Calculates the number of tokens in a text using tiktoken

### Constants: `1min-relay/utils/constants.py`
- `ENDPOINTS`: Dictionary of API endpoints
- `ROLES_MAPPING`: Mapping of roles for different models
- `MODEL_CAPABILITIES`: Dictionary of model capabilities
- Various other constants used throughout the application

### Imports: `1min-relay/utils/imports.py`
- Central place for all standard library imports
- Imports used across multiple modules

### Logger: `1min-relay/utils/logger.py`
- `logger`: Configured logging instance
- Functions for setting up and using the logger

### Memcached: `1min-relay/utils/memcached.py`
- `MEMORY_STORAGE`: Dictionary for temporary storage
- `safe_memcached_operation`: Safely performs operations on memcached
- `delete_all_files_task`: Periodically cleans up outdated user files

## Functions

### Functions initialization: `1min-relay/routes/functions/__init__.py`
- Clear export of all necessary functions from submodules
- Grouping and documenting functions by categories
- Provides convenient import of functions in routes

### Shared functions: `1min-relay/routes/functions/shared_func.py`
- `validate_auth`: Validates the authorization header
- `handle_api_error`: Standardized error handling for API responses
- `format_openai_response`: Formats responses to match OpenAI API
- `format_image_response`: Formats image responses to match OpenAI API
- `stream_response`: Streams API responses
- `get_full_url`: Creates a full URL from a relative path
- `extract_data_from_api_response`: Common function for extracting data from API responses
- `extract_text_from_response`: Extracts text from API responses
- `extract_image_urls`: Extracts image URLs from API responses
- `extract_audio_url`: Extracts audio URL from API responses

### Text functions: `1min-relay/routes/functions/txt_func.py`
- `format_conversation_history`: Formats conversation history for models
- `get_model_capabilities`: Gets capability information for a model
- `prepare_payload`: Prepares the payload for API requests
- `transform_response`: Transforms API responses
- `emulate_stream_response`: Emulates a streaming response

### Image functions: `1min-relay/routes/functions/img_func.py`
- `build_generation_payload`: Builds the payload for image generation
- `parse_aspect_ratio`: Parses the aspect ratio from input
- `create_image_variations`: Creates variations of images
- `build_img2img_payload`: Builds payload for img2img requests
- `process_image_tool_calls`: Processes image tool calls

### Audio functions: `1min-relay/routes/functions/audio_func.py`
- `upload_audio_file`: Uploads audio files
- `try_models_in_sequence`: Tries different models sequentially
- `prepare_models_list`: Prepares a list of models to try
- `prepare_whisper_payload`: Prepares payload for Whisper API
- `prepare_tts_payload`: Prepares payload for text-to-speech

### File functions: `1min-relay/routes/functions/file_func.py`
- `get_user_files`: Gets user files from Memcached
- `save_user_files`: Saves user files to Memcached
- `create_temp_file`: Creates temporary files
- `upload_asset`: Uploads assets to the server
- `get_mime_type`: Gets the MIME type of a file
- `retry_image_upload`: Retries image upload on failure
- `format_file_response`: Formats file response in OpenAI format
- `create_api_response`: Creates HTTP response with proper headers
- `find_conversation_id`: Finds conversation ID in API response
- `find_file_by_id`: Finds file by ID in user's files list
- `create_conversation_with_files`: Creates a new conversation with files

## Routes

### Text routes: `1min-relay/routes/text.py`
- `/v1/models`: Returns a list of available models
- `/v1/chat/completions`: Handles chat completion requests
- Various other text model endpoints

### Image routes: `1min-relay/routes/images.py`
- `/v1/images/generations`: Generates images from text
- `/v1/images/variations`: Creates variations of images

### Audio routes: `1min-relay/routes/audio.py`
- `/v1/audio/transcriptions`: Transcribes audio to text
- `/v1/audio/translations`: Translates audio to another language
- `/v1/audio/speech`: Converts text to speech

### File routes: `1min-relay/routes/files.py`
- `/v1/files`: Handles file upload and management
- `/v1/files/<file_id>`: Gets or deletes a specific file
- `/v1/files/<file_id>/content`: Gets file content 
