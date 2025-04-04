from flask import Flask, request, jsonify, make_response, redirect
from flask_cors import CORS, cross_origin
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from waitress import serve
# другие импорты...

# Глобальные переменные и инициализация
app = Flask(__name__)
CORS(app)
limiter = Limiter(...)
PORT = 5000
# другие глобальные переменные...

# Основные настройки
if __name__ == "__main__":
    # Launch the task of deleting files
    delete_all_files_task()

    # Логирование старта сервера
    # ...

    # Запуск сервера
    serve(app, host="0.0.0.0", port=PORT, threads=6)
