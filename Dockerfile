FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-memcached

COPY . .

# Создаем директорию для временных файлов
RUN mkdir -p temp

EXPOSE 5001

# Изменяем запуск с main.py на app.py
CMD ["python", "app.py"]
