# 1min-relay

## Project Description
1min-relay is a proxy server implementing an API compatible with the OpenAI API for working with various AI models through the 1min.ai service. It allows you to use client applications that support the OpenAI API with models from various providers through a unified interface.

## Features
- Fully compatible with the OpenAI API, including chat/completions, images, audio, and files
- Added an OpenAI-compatible **`POST /v1/responses`** endpoint (best-effort) for clients/SDKs that use the Responses API
- Supports a large number of models from various providers: OpenAI, Claude, Mistral, Google, and others
- Works with different types of requests: text, images, audio, and files
- Implements data streaming
- Has a rate limiting function using Memcached
- Allows you to set a subset of allowed models through environment variables
- **Best-effort dynamic `/v1/models`**: when an API key is provided, the server can fetch a live model list from upstream with caching (otherwise falls back to the static list)
- **Graceful web_search degradation**: if upstream returns `400` when webSearch is enabled, the proxy retries once with webSearch disabled and sets `X-WebSearch-Degraded: true`
- **OpenClaw tool-calling (emulation, best-effort)**: tool calling is enabled **only** for OpenClaw requests (by headers). For other clients, `tools` are ignored so streaming is not impacted.
- Optimized modular structure with minimal code duplication

## 1min.ai API Notes (Upstream)
This proxy converts OpenAI-like requests into the **current** 1min.ai API:

- **Chat**: uses [Chat with AI API](https://docs.1min.ai/docs/api/chat-with-ai-api)
  - `POST https://api.1min.ai/api/chat-with-ai`
  - `POST https://api.1min.ai/api/chat-with-ai?isStreaming=true` (SSE)
- **Non-chat features** (images/audio/etc.): use [AI Feature API](https://docs.1min.ai/docs/api/ai-feature-api)
- **Assets/files upload**: use [Asset API](https://docs.1min.ai/docs/api/asset-api)

Upstream authentication is performed with the `API-KEY` header. This server accepts both:
- `Authorization: Bearer <YOUR_1MIN_API_KEY>` (recommended for OpenAI clients)
- `API-KEY: <YOUR_1MIN_API_KEY>`

## Project Structure
The project has a modular structure to facilitate development and maintenance:

```
1min-relay/
├── app.py                # Main application file - server initialization and settings
├── utils/                # Common utilities and modules
│   ├── __init__.py       # Package initialization
│   ├── common.py         # Common helper functions
│   ├── constants.py      # Constants and configuration variables
│   ├── imports.py        # Centralized imports
│   ├── logger.py         # Logging setup
│   └── memcached.py      # Functions for working with Memcached
├── routes/               # API routes
│   ├── __init__.py       # Routes module initialization
│   ├── text.py           # Routes for text requests
│   ├── images.py         # Routes for working with images
│   ├── audio.py          # Routes for audio requests
│   ├── files.py          # Routes for working with files
│   └── functions/        # Helper functions for different types of requests
│       ├── __init__.py   # Functions package initialization
│       ├── shared_func.py# Common helper functions for all request types
│       ├── txt_func.py   # Helper functions for text models
│       ├── img_func.py   # Helper functions for working with images
│       ├── audio_func.py # Helper functions for working with audio
│       └── file_func.py  # Helper functions for working with files
├── requirements.txt      # Project dependencies
├── INSTALL.sh            # Local installation script (venv)
├── RUN.sh                # Local launch script (venv)
├── UPDATE.sh             # Docker container update script
├── Dockerfile            # Instructions for building Docker image
├── CODE_STRUCTURE.md     # Detailed information about code structure
└── README.md             # Project documentation
```

### Key Components:

- **app.py**: The main application file that initializes the server, configures settings, and sets up the Flask application.

- **utils/**: Contains essential utility modules that provide core functionality:
  - common.py: Common helper functions used throughout the application
  - constants.py: Defines all constants, configuration variables, and model lists
  - imports.py: Centralizes imports to avoid circular dependencies
  - logger.py: Configures logging for the application
  - memcached.py: Provides rate limiting functionality

- **routes/**: Contains the main API endpoints that implement the OpenAI API compatibility:
  - text.py: Implements chat/completions endpoints
  - images.py: Implements image generation and processing endpoints
  - audio.py: Implements speech-to-text and text-to-speech endpoints
  - files.py: Implements file management endpoints

- **routes/functions/**: Contains helper functions that support the main route handlers:
  - shared_func.py: Common helper functions for all request types
  - txt_func.py: Helper functions for text models
  - img_func.py: Helper functions for working with images
  - audio_func.py: Helper functions for working with audio
  - file_func.py: Helper functions for working with files

## Requirements
- Python 3.7+
- Flask and related libraries
- Memcached (optional for rate limiting)
- 1min.ai service API key

## Installation and Launch

### Installing Dependencies
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
```
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables Setup
Create a `.env` file in the project root directory:
```
PORT=5001
SUBSET_OF_ONE_MIN_PERMITTED_MODELS=gpt-4o-mini,mistral-nemo,claude-3-haiku-20240307,gemini-1.5-flash
PERMIT_MODELS_FROM_SUBSET_ONLY=false
```

### Server Launch
```bash
sudo apt install memcached libmemcached-tools -y
sudo systemctl enable memcached
sudo systemctl start memcached
```
```bash
source venv/bin/activate
python app.py
```
After launching, the server will be available at `http://localhost:5001/`.

### Scripts for Automating Local Installation (venv), Local Launch (venv), Update (Docker container)

```bash
chmod +x *.sh
# python-venv
sudo ./INSTALL.sh
./RUN.sh
# docker (reinstall)
mv UPDATE.sh ../
cd ../
./UPDATE.sh
```

## Usage with OpenAI API Clients
Most OpenAI API clients can be configured to use this server by specifying the base URL:
```
http://localhost:5001/v1
```

When sending requests to the API, use your 1min.ai API key in the Authorization header (OpenAI-compatible):
```
Authorization: Bearer your-1min-api-key
```

### Streaming
For `stream: true` on `/v1/chat/completions`, the server will stream responses as **OpenAI-style SSE** (`data: {...}\n\n` + `data: [DONE]`),
while consuming the upstream 1min.ai SSE events (`event: content/result/done/error`).

### OpenClaw: tool calling (emulation, best-effort)
1min.ai `UNIFY_CHAT_WITH_AI` does not reliably provide native OpenAI `tool_calls`, so for OpenClaw this proxy implements **tool-calling emulation**:

- **Detect OpenClaw client**: via `X-OpenClaw: true` and/or `X-Client: openclaw` (optionally `User-Agent` containing `openclaw`)
- **Do not affect other clients**: for non-OpenClaw requests, `tools`, `tool_choice`, `parallel_tool_calls` are **dropped/ignored** so streaming stays enabled and behavior remains OpenAI-client friendly
- **Streaming stays on by default** (including OpenClaw)
- **Per-response streaming downgrade**: when OpenClaw sends `tools` (function), the proxy performs a single non-stream “probe” upstream call
  - if `tool_calls` are found (as JSON embedded in assistant text), the proxy returns a **non-streaming** OpenAI-like response with `finish_reason="tool_calls"` and `message.tool_calls`
  - if no `tool_calls` are found, the proxy returns an **emulated SSE stream** from the full assistant content

Note: robust upstream text extraction was added because 1min.ai can place the assistant text under different fields depending on the model/endpoint. This prevents “empty” completions (0 completion tokens) that would otherwise block OpenClaw file/memory writes (e.g. `MEMORY.md`, `memory/*.md`).

### Responses API (best-effort)
The server also supports `POST /v1/responses` (non-streaming). Example:

```bash
curl http://localhost:5001/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-1min-api-key" \
  -d '{
    "model": "gpt-4o-mini",
    "input": "Return a JSON object with ok=true",
    "response_format": { "type": "json_object" }
  }'
```

## Launching Using Docker
You can also run the server in a Docker container:

```bash
    docker run -d --name 1min-relay-container --restart always --network 1min-relay-network -p 5001:5001 \
      -e SUBSET_OF_ONE_MIN_PERMITTED_MODELS="mistral-nemo,gpt-4o-mini,deepseek-chat" \
      -e PERMIT_MODELS_FROM_SUBSET_ONLY=False \
      -e MEMCACHED_HOST=memcached \
      -e MEMCACHED_PORT=11211 \
      1min-relay-container:latest
```

## License
[MIT License](LICENSE) 
