#!/bin/bash
#########################
# MOVE ME TO 1 LEVEL UP #
#=======================#
# mv UPDATE.sh ../      #
# cd ../                #
# chmod +x UPDATE.sh    #
# ./UPDATE.sh           #
#########################
set -eu

# Удаление старой папки
rm -rf ./1min-relay/

# Клонирование нового репозитория
#git clone https://github.com/kokofixcomputers/1min-relay.git ./1min-relay
git clone https://github.com/chelaxian/1min-relay.git ./1min-relay
#git clone -b test https://github.com/chelaxian/1min-relay.git ./1min-relay

# Переход в директорию
cd ./1min-relay
chmod +x UPDATE.sh
cp UPDATE.sh ../

# Проверяем, что мы находимся в директории проекта
if [ ! -f "app.py" ]; then
    echo "Ошибка: скрипт должен быть запущен из директории проекта 1min-relay"
    exit 1
fi

# Проверка и создание Docker сети, если она не существует
if ! docker network ls | grep -q "1min-relay-network"; then
    echo "Создание Docker сети '1min-relay-network'..."
    docker network create 1min-relay-network
fi

# Остановка и удаление контейнеров, если они существуют
if docker ps -a | grep -q "1min-relay-container"; then
    echo "Остановка и удаление контейнера 1min-relay-container..."
    docker stop 1min-relay-container || true
    docker rm 1min-relay-container || true
fi

if docker ps -a | grep -q "memcached"; then
    echo "Остановка и удаление контейнера memcached..."
    docker stop memcached || true
    docker rm memcached || true
fi

# Удаление старого образа
if docker images | grep -q "1min-relay-container"; then
    echo "Удаление старого образа 1min-relay-container..."
    docker rmi 1min-relay-container:latest || true
fi

# Проверка наличия docker-compose.yml и запуск через него
#if [ -f "docker-compose.yml" ]; then
#    echo "Запуск через docker-compose..."
#    docker-compose up -d
#else
    # Если docker-compose.yml не найден, используем обычный Docker
    echo "Запуск через Dockerfile..."
    
    # Сборка нового образа из локального проекта
    docker build -t 1min-relay-container:latest .
    
    # Запуск memcached
    docker run -d --name memcached --restart always --network 1min-relay-network memcached:latest
    
    # Запуск контейнера с политикой перезапуска
    docker run -d --name 1min-relay-container --restart always --network 1min-relay-network -p 5001:5001 \
      -e SUBSET_OF_ONE_MIN_PERMITTED_MODELS="mistral-nemo,gpt-4o-mini,deepseek-chat" \
      -e PERMIT_MODELS_FROM_SUBSET_ONLY=False \
      -e MEMCACHED_HOST=memcached \
      -e MEMCACHED_PORT=11211 \
      1min-relay-container:latest
#fi

echo "1min-relay-container успешно обновлен и запущен!"

# Функция для отображения логов контейнера
show_logs() {
    echo "Выводим логи контейнера 1min-relay-container (Ctrl+C для выхода)..."
    docker logs -f 1min-relay-container
}

# Функция для входа в консоль контейнера
enter_console() {
    echo "Входим в консоль контейнера 1min-relay-container..."
    docker exec -it 1min-relay-container /bin/bash
}

# Даем пользователю выбор действия
echo ""
echo "Выберите действие:"
echo "1) Просмотр логов контейнера в реальном времени"
echo "2) Вход в консоль контейнера"
echo "3) Выход из скрипта"
read -p "Ваш выбор (1-3): " choice

case $choice in
    1)
        show_logs
        ;;
    2)
        enter_console
        ;;
    3)
        echo "Выход из скрипта."
        exit 0
        ;;
    *)
        echo "Неверный выбор. Выход из скрипта."
        exit 1
        ;;
esac
