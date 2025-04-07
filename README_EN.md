# 1min-relay

## Project Description
1min-relay is a proxy server implementing an API compatible with the OpenAI API for working with various AI models through the 1min.ai service. It allows you to use client applications that support the OpenAI API with models from various providers through a unified interface.

## Features
- Fully compatible with the OpenAI API, including chat/completions, images, audio, and files
- Supports a large number of models from various providers: OpenAI, Claude, Mistral, Google, and others
- Works with different types of requests: text, images, audio, and files
- Implements data streaming
- Has a rate limiting function using Memcached
- Allows you to set a subset of allowed models through environment variables
- Optimized modular structure with minimal code duplication

## Project Structure
The project has a modular structure to facilitate development and maintenance:

```
1min-relay/
├── app.py                # Main application file
├── utils/                # Common utilities and modules
│   ├── __init__.py       # Package initialization
│   ├── common.py         # Common helper functions
│   ├── constants.py      # Constants and configuration variables
│   ├── imports.py        # Centralized imports
│   ├── logger.py         # Logging setup
│   └── memcached.py      # Functions for working with Memcached
├── routes/               # API routes
│   ├── __init__.py       # Routes module initialization
│   ├── functions/        # Functions for different types of requests
│   │   ├── __init__.py   # Functions package initialization with clear exports
│   │   ├── shared_func.py# Common functions for all request types
│   │   │                 # (authentication, error handling, response formatting,
│   │   │                 # extraction of data from API responses)
│   │   ├── txt_func.py   # Functions for text models
│   │   ├── img_func.py   # Functions for working with images
│   │   ├── audio_func.py # Functions for working with audio
│   │   └── file_func.py  # Functions for working with files
│   ├── text.py           # Routes for text requests
│   ├── images.py         # Routes for working with images
│   ├── audio.py          # Routes for audio requests
│   └── files.py          # Routes for working with files
├── requirements.txt      # Project dependencies
├── INSTALL.sh            # Local installation script (venv)
├── RUN.sh                # Local launch script (venv)
├── UPDATE.sh             # Docker container update script
├── Dockerfile            # Instructions for building Docker image
├── CODE_STRUCTURE.md     # Detailed information about code structure
└── README.md             # Project documentation
```

### Key Components:

- **shared_func.py**: Contains common functions used in all types of routes:
  - Authorization and error handling
  - Response formatting to match the OpenAI API
  - Universal functions for extracting data from API responses
  - Common code for working with URLs and multimedia content

- **Specialized function modules**: Contain only unique logic for specific types of requests:
  - txt_func.py: Processing text requests
  - img_func.py: Image generation and processing
  - audio_func.py: Audio transcription and speech synthesis
  - file_func.py: File management

- **API Routes**: Implement endpoints compatible with the OpenAI API using the corresponding processing functions.

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

When sending requests to the API, use your 1min.ai API key in the Authorization header:
```
Authorization: Bearer your-1min-api-key
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
