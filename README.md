# 1min-Relay
Relay 1min AI API to OpenAI Structure in 1 minute.

Don't forget to star this repository if you like it! 

Oh and by the way, join our discord server for support and updates!
[https://discord.gg/GQd3DrxXyj](https://discord.gg/GQd3DrxXyj)

## Features
- **bolt.diy Support**: Compatible with bolt.diy for seamless integration.
- **Conversation History**: Retain and manage conversation history effortlessly.
- **Broad Client Compatibility**: Works with most clients supporting OpenAI Custom Endpoint.
- **Fast and Reliable Relay**: Relays 1min AI API to OpenAI-compatible structure within 1 minute.
- **User-Friendly**: Easy to set up and quick to use.
- **Model Exposure Control**: Expose all or a predefined subset of 1min.ai-supported models.
- **Streaming Support**: Enables real-time streaming for faster interactions.
- **Non-Streaming Support**: Compatible with non-streaming workflows.
- **Docker Support**: Deploy easily with Docker for fast and consistent setup.
- **Multi-Document Support**: Supports document uploads for enhanced functionality. Some include .docx, .pdf, .txt, .yaml and etc.
- **Image Support**: Supports image uploads for enhanced functionality.
- **ARM64 and AMD64 Support**: Compatible with both ARM64 and AMD64 architectures.
- **Multiple Requests**: Supports multiple requests simultaneously.


## Installation:

Clone the git repo into your machine with:
```bash
git clone https://github.com/kokofixcomputers/1min-relay.git
```

### Bare-metal

To install dependencies, run:
```bash
pip install -r requirements.txt
```

and then run python3 main.py

Depending on your system, you may need to run `python` instead of `python3`.

### Docker
Running 1min-relay in docker is the easiet method. Please note that the connection ip address displayed on server start will be wrong when in docker.

#### Pre-Built images

1. Pull the Docker Image:
```bash
docker pull kokofixcomputers/1min-relay:latest
```

2. To encrease security, 1min-relay will require it's own network to be able to communicate with memcached.
To create a network, run:
```bash
docker network create 1min-relay-network
```

3. Run Memcached.
```bash
docker run -d --name memcached --network 1min-relay-network memcached
```

4. Run the 1min-relay Container:
```bash
docker run -d --name 1min-relay-container --network 1min-relay-network -p 5001:5001 \
  -e SUBSET_OF_ONE_MIN_PERMITTED_MODELS="mistral-nemo,gpt-4o-mini,deepseek-chat" \
  -e PERMIT_MODELS_FROM_SUBSET_ONLY=True \
  kokofixcomputers/1min-relay:latest
```
Environment Variables:

- `SUBSET_OF_ONE_MIN_PERMITTED_MODELS`: Specifies a subset of 1min.ai models to expose. Default: mistral-nemo,gpt-4o,deepseek-chat.
- `PERMIT_MODELS_FROM_SUBSET_ONLY`: Restricts model usage to the specified subset. Set to True to enforce this restriction or False to allow all models supported by 1min.ai. Default: True.


#### Self-Build

1. Build the Docker Image
From the project directory (where Dockerfile and main.py reside), run:

```bash
docker build -t 1min-relay:latest .
```

2. To encrease security, 1min-relay will require it's own network to be able to communicate with memcached.
To create a network, run:
```bash
docker network create 1min-relay-network
```

3. Run Memcached.
```bash
docker run -d --name memcached --network 1min-relay-network memcached
```

4. Run the 1min-relay Container:
```bash
docker run -d --name 1min-relay-container --network 1min-relay-network -p 5001:5001 \
  -e SUBSET_OF_ONE_MIN_PERMITTED_MODELS="mistral-nemo,gpt-4o-mini,deepseek-chat" \
  -e PERMIT_MODELS_FROM_SUBSET_ONLY=True \
  1min-relay:latest
```

- `-d` runs the container in detached (background) mode.
- `-p 5001:5001` maps your host’s port 5001 to the container’s port 5001.
- `--name 1min-relay-container` is optional, but it makes it easier to stop or remove the container later.
- `-e`: Specifies environment variables.
- `SUBSET_OF_ONE_MIN_PERMITTED_MODELS`: Specifies a subset of 1min.ai models to expose. Default: mistral-nemo,gpt-4o,deepseek-chat.
- `PERMIT_MODELS_FROM_SUBSET_ONLY`: Restricts model usage to the specified subset. Set to True to enforce this restriction or False to allow all models supported by 1min.ai. Default: False.


4. Verify It’s Running
Check logs (optional):

```bash
docker logs -f 1min-relay-container
```
You should see your Flask server output: “Server is ready to serve at …”

Test from your host machine:

```bash
curl -X GET http://localhost:5001/v1/models
```

5. Stopping or Removing the Container
To stop the container:

```bash
docker stop 1min-relay-container
```

To remove the container:

```bash
docker rm 1min-relay-container
```

To remove the image entirely:

```bash
docker rmi 1min-relay:latest
```

Optional: Run with Docker Compose
If you prefer Docker Compose, you can run the docker compose included with the repo:

Just run:

```bash
docker compose up -d
```
Compose will automatically do these things for you:
- Create a network
- Run Memcached
- Run the 1min-relay Container

#### Managing docker containers
You can also have multiple instances of 1min-relay running on the same machine with docker.

You just need to chznge the name parameter of the container so you can identify them.

To stop a 1min-relay container, run:
```bash
docker stop 1min-relay-container
```
To start it again, run:
```bash
docker start 1min-relay-container
```

### Environment Variables
- `SUBSET_OF_ONE_MIN_PERMITTED_MODELS`: Specifies a subset of 1min.ai models to expose. Default: mistral-nemo,gpt-4o,deepseek-chat.
- `PERMIT_MODELS_FROM_SUBSET_ONLY`: Restricts model usage to the specified subset. Set to True to enforce this restriction or False to allow all models supported by 1min.ai. Default: False.
