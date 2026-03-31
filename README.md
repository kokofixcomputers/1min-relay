# 1min-relay

## Описание проекта
1min-relay - это сервер-посредник (прокси), реализующий API, совместимый с OpenAI API, для работы с различными AI-моделями через сервис 1min.ai. Он позволяет использовать клиентские приложения, поддерживающие OpenAI API, с моделями различных провайдеров через единый интерфейс.

## Особенности
- Полностью совместим с OpenAI API, включая chat/completions, images, audio и files
- Добавлен OpenAI-совместимый endpoint **`POST /v1/responses`** (best-effort) для клиентов/SDK, которые используют Responses API
- Поддерживает большое количество моделей от различных провайдеров: OpenAI, Claude, Mistral, Google и других
- Работает с различными типами запросов: текстовыми, изображениями, аудио и файлами
- Реализует потоковую передачу данных (streaming)
- Имеет функцию ограничения запросов (rate limiting) с использованием Memcached
- Позволяет задать подмножество разрешенных моделей через переменные окружения
- **Динамический `/v1/models` (best-effort)**: при наличии API-ключа может подтягивать живой список моделей из upstream и кэшировать (иначе использует статический список)
- **Graceful degradation web_search**: если upstream возвращает `400` при включенном webSearch, прокси повторяет запрос без webSearch и ставит заголовок `X-WebSearch-Degraded: true`
- **OpenClaw tool-calling (эмуляция, best-effort)**: tool calling включается **только** для запросов от OpenClaw (по заголовкам), для остальных клиентов `tools` игнорируются, и стриминг не ломается
- Оптимизированная модульная структура с минимальным дублированием кода

## Примечания по upstream 1min.ai API
Прокси конвертирует OpenAI-like запросы в **актуальный** API 1min.ai:

- **Чат**: [Chat with AI API](https://docs.1min.ai/docs/api/chat-with-ai-api)
  - `POST https://api.1min.ai/api/chat-with-ai`
  - `POST https://api.1min.ai/api/chat-with-ai?isStreaming=true` (SSE)
- **Не-чат фичи** (генерация/вариации изображений, аудио и т.д.): [AI Feature API](https://docs.1min.ai/docs/api/ai-feature-api)
- **Загрузка файлов/картинок**: [Asset API](https://docs.1min.ai/docs/api/asset-api)

Upstream 1min.ai использует заголовок `API-KEY`. Этот сервер принимает оба варианта:
- `Authorization: Bearer <ВАШ_1MIN_API_KEY>` (рекомендуется для OpenAI-клиентов)
- `API-KEY: <ВАШ_1MIN_API_KEY>`

## Структура проекта
Проект имеет модульную структуру для облегчения разработки и поддержки:

```
1min-relay/
├── app.py                # Основной файл приложения - инициализация сервера и настройки
├── utils/                # Общие утилиты и модули
│   ├── __init__.py       # Инициализация пакета
│   ├── common.py         # Общие вспомогательные функции
│   ├── constants.py      # Константы и конфигурационные переменные
│   ├── imports.py        # Централизованные импорты
│   ├── logger.py         # Настройка логирования
│   └── memcached.py      # Функции для работы с Memcached
├── routes/               # Маршруты API
│   ├── __init__.py       # Инициализация модуля маршрутов
│   ├── text.py           # Маршруты для текстовых запросов
│   ├── images.py         # Маршруты для работы с изображениями
│   ├── audio.py          # Маршруты для аудио запросов
│   ├── files.py          # Маршруты для работы с файлами
│   └── functions/        # Вспомогательные функции для различных типов запросов
│       ├── __init__.py   # Инициализация пакета функций
│       ├── shared_func.py# Общие вспомогательные функции для всех типов запросов
│       ├── txt_func.py   # Вспомогательные функции для текстовых моделей
│       ├── img_func.py   # Вспомогательные функции для работы с изображениями
│       ├── audio_func.py # Вспомогательные функции для работы с аудио
│       └── file_func.py  # Вспомогательные функции для работы с файлами
├── requirements.txt      # Зависимости проекта
├── INSTALL.sh            # Скрипт локальной установки (venv)
├── RUN.sh                # Скрипт локального запуска (venv)
├── UPDATE.sh             # Скрипт обновления Docker-контейнера
├── Dockerfile            # Инструкции для сборки Docker-образа
├── CODE_STRUCTURE.md     # Подробная информация о структуре кода
└── README.md             # Документация проекта
```

### Ключевые компоненты:

- **app.py**: Основной файл приложения, который инициализирует сервер, настраивает параметры и создает Flask-приложение.

- **utils/**: Содержит основные утилитные модули, обеспечивающие базовую функциональность:
  - common.py: Общие вспомогательные функции, используемые во всем приложении
  - constants.py: Определяет все константы, конфигурационные переменные и списки моделей
  - imports.py: Централизует импорты для избежания циклических зависимостей
  - logger.py: Настраивает логирование для приложения
  - memcached.py: Обеспечивает функциональность ограничения запросов

- **routes/**: Содержит основные API-эндпоинты, реализующие совместимость с OpenAI API:
  - text.py: Реализует эндпоинты chat/completions
  - images.py: Реализует эндпоинты для генерации и обработки изображений
  - audio.py: Реализует эндпоинты для преобразования речи в текст и текста в речь
  - files.py: Реализует эндпоинты для управления файлами

- **routes/functions/**: Содержит вспомогательные функции, поддерживающие основные обработчики маршрутов:
  - shared_func.py: Общие вспомогательные функции для всех типов запросов
  - txt_func.py: Вспомогательные функции для текстовых моделей
  - img_func.py: Вспомогательные функции для работы с изображениями
  - audio_func.py: Вспомогательные функции для работы с аудио
  - file_func.py: Вспомогательные функции для работы с файлами

## Требования
- Python 3.7+
- Flask и связанные библиотеки
- Memcached (опционально для rate limiting)
- API ключ сервиса 1min.ai

## Установка и запуск

### Установка зависимостей
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

### Настройка переменных окружения
Создайте файл `.env` в корневой директории проекта:
```
PORT=5001
SUBSET_OF_ONE_MIN_PERMITTED_MODELS=gpt-4o-mini,mistral-nemo,claude-3-haiku-20240307,gemini-1.5-flash
PERMIT_MODELS_FROM_SUBSET_ONLY=false
```

### Запуск сервера
```bash
sudo apt install memcached libmemcached-tools -y
sudo systemctl enable memcached
sudo systemctl start memcached
```
```bash
source venv/bin/activate
python app.py
```
После запуска сервер будет доступен по адресу `http://localhost:5001/`.

### Скрипты для автоматизации локальной установки (venv), локального запуска (venv), обновления (Docker-контейнер)

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

## Использование с клиентами OpenAI API
Большинство клиентов OpenAI API могут быть настроены для использования этого сервера путем указания базового URL:
```
http://localhost:5001/v1
```

При отправке запросов к API используйте свой API ключ 1min.ai в заголовке Authorization (OpenAI-совместимо):
```
Authorization: Bearer your-1min-api-key
```

### Потоковый режим (streaming)
Если вы передаёте `stream: true` в `/v1/chat/completions`, сервер вернёт **OpenAI-style SSE** (`data: {...}\n\n` + `data: [DONE]`),
а upstream 1min.ai будет потребляться как SSE-события `event: content/result/done/error`.

### OpenClaw: tool calling (эмуляция, best-effort)
1min.ai в режиме `UNIFY_CHAT_WITH_AI` не гарантирует нативный OpenAI tool-calling формат, поэтому для OpenClaw реализована **эмуляция**:

- **Распознавание OpenClaw-клиента**: по HTTP-заголовкам `X-OpenClaw: true` и/или `X-Client: openclaw` (также допускается `User-Agent` содержащий `openclaw`)
- **Изоляция от остальных клиентов**: для не-OpenClaw запросов поля `tools`, `tool_choice`, `parallel_tool_calls` **игнорируются/удаляются**, чтобы не отключать стриминг и не ломать поведение обычных OpenAI-клиентов
- **Стриминг по умолчанию сохраняется**: даже для OpenClaw
- **Только для одного ответа стрим может быть “погашен”**: если OpenClaw прислал `tools` (function), сервер делает 1 “probe” нестриминговый запрос upstream
  - если в ответе обнаружены `tool_calls` (в виде JSON в тексте) — возвращается **нестриминговый** OpenAI-like ответ с `finish_reason="tool_calls"` и полем `message.tool_calls`
  - если `tool_calls` не обнаружены — возвращается **эмулированный стрим** (SSE) из полного текста ответа

Примечание: был добавлен “robust” парсинг текста ответа upstream, т.к. 1min.ai может возвращать результат в разных полях. Это устраняет ситуацию, когда ответ приходил пустым (0 completion tokens), и OpenClaw не мог выполнить запись `MEMORY.md`/`memory/*.md`.

### Responses API (best-effort)
Также поддерживается `POST /v1/responses` (нестриминговый). Пример запроса:

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

## Запуск с использованием Docker
Вы также можете запустить сервер в Docker-контейнере:

```bash
    docker run -d --name 1min-relay-container --restart always --network 1min-relay-network -p 5001:5001 \
      -e SUBSET_OF_ONE_MIN_PERMITTED_MODELS="mistral-nemo,gpt-4o-mini,deepseek-chat" \
      -e PERMIT_MODELS_FROM_SUBSET_ONLY=False \
      -e MEMCACHED_HOST=memcached \
      -e MEMCACHED_PORT=11211 \
      1min-relay-container:latest
```

## Лицензия
[MIT License](LICENSE)
