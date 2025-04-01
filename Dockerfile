FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-memcached

COPY . .

EXPOSE 5001

CMD ["python", "main.py"]
