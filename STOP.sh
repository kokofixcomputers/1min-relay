#!/bin/bash
#########################
# run with sudo

# stop app.py
killall -9 python3
killall -9 python
# stop memcached
killall memcached
systemctl kill memcached
systemctl stop memcached

#########################
# Остановка контейнеров, если они существуют
if docker ps -a | grep -q "1min-relay-container"; then
    echo "Остановка контейнера 1min-relay-container..."
    docker stop 1min-relay-container || true
fi

if docker ps -a | grep -q "memcached"; then
    echo "Остановка контейнера memcached..."
    docker stop memcached || true
fi

echo "1min-relay остановлен!"
