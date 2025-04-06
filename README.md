# 1min-relay

## Описание проекта
1min-relay - это сервер-посредник (прокси), реализующий API, совместимый с OpenAI API, для работы с различными AI-моделями через сервис 1min.ai. Он позволяет использовать клиентские приложения, поддерживающие OpenAI API, с моделями различных провайдеров через единый интерфейс.

## Особенности
- Полностью совместим с OpenAI API, включая chat/completions, images, audio и files
- Поддерживает большое количество моделей от различных провайдеров: OpenAI, Claude, Mistral, Google и других
- Работает с различными типами запросов: текстовыми, изображениями, аудио и файлами
- Реализует потоковую передачу данных (streaming)
- Имеет функцию ограничения запросов (rate limiting) с использованием Memcached
- Позволяет задать подмножество разрешенных моделей через переменные окружения

## Структура проекта
Проект имеет модульную структуру для облегчения разработки и поддержки:

```
1min-relay/
├── app.py                # Основной файл приложения
├── utils/                # Общие утилиты и модули
│   ├── __init__.py       # Инициализация пакета
│   ├── common.py         # Общие вспомогательные функции
│   ├── constants.py      # Константы и конфигурационные переменные
│   ├── imports.py        # Централизованные импорты
│   ├── logger.py         # Настройка логирования
│   └── memcached.py      # Функции для работы с Memcached
├── routes/               # Маршруты API
│   ├── __init__.py       # Инициализация модуля маршрутов
│   ├── utils.py          # Общие функции для маршрутов
│   ├── text.py           # Маршруты для текстовых запросов
│   ├── images.py         # Маршруты для работы с изображениями
│   ├── audio.py          # Маршруты для аудио запросов
│   └── files.py          # Маршруты для работы с файлами
├── requirements.txt      # Зависимости проекта
├── docker-compose.yml    # Конфигурация Docker Compose
├── Dockerfile            # Инструкции для сборки Docker-образа
├── LICENSE               # Лицензия проекта
├── INFO.md               # Подробная информация о структуре кода
└── README.md             # Документация проекта
```

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
python3 -m venv ven
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
python app.py
```

После запуска сервер будет доступен по адресу `http://localhost:5001/`.

## Использование с клиентами OpenAI API
Большинство клиентов OpenAI API могут быть настроены для использования этого сервера путем указания базового URL:
```
http://localhost:5001/v1
```

При отправке запросов к API используйте свой API ключ 1min.ai в заголовке Authorization:
```
Authorization: Bearer your-1min-api-key
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
