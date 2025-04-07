# routes/async_tasks.py

import threading
import time
import uuid
import json
import os
import re
from datetime import datetime, timedelta

from utils.imports import *
from utils.logger import logger
from utils.constants import *
from utils.common import (
    api_request, 
    create_session
)
from utils.memcached import safe_memcached_operation
from routes.functions.shared_func import extract_image_urls, get_full_url

# Словарь для хранения информации о задачах
TASKS = {}
TASKS_LOCK = threading.Lock()

# Очистка старых задач каждые 24 часа
def cleanup_old_tasks():
    with TASKS_LOCK:
        current_time = datetime.now()
        tasks_to_remove = []
        
        for task_id, task_info in TASKS.items():
            # Удаляем задачи старше 24 часов
            if current_time - task_info.get('created_at', current_time) > timedelta(hours=24):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del TASKS[task_id]
            logger.info(f"Очищена устаревшая задача {task_id}")
    
    # Запускаем следующую очистку через 24 часа
    timer = threading.Timer(86400, cleanup_old_tasks)
    timer.daemon = True
    timer.start()

# Запускаем таймер очистки при загрузке модуля
cleanup_timer = threading.Timer(86400, cleanup_old_tasks)
cleanup_timer.daemon = True
cleanup_timer.start()

def process_image_generation_async(task_id, api_key, payload, model, prompt, request_id):
    """
    Асинхронная обработка запроса на генерацию изображения
    
    Args:
        task_id: Идентификатор задачи
        api_key: API ключ пользователя
        payload: Полезная нагрузка для API запроса
        model: Название модели
        prompt: Текст подсказки
        request_id: Идентификатор запроса
    """
    with TASKS_LOCK:
        TASKS[task_id]['status'] = 'processing'
    
    logger.info(f"[{request_id}] Начата асинхронная обработка задачи {task_id} для модели {model}")
    
    try:
        # Создаем сессию для запросов
        session = create_session()
        headers = {"API-KEY": api_key, "Content-Type": "application/json"}
        api_url = f"{ONE_MIN_API_URL}"
        
        # Увеличенный таймаут для Midjourney
        timeout = 300 if model in ["midjourney", "midjourney_6_1"] else 180
        
        # Выполняем запрос к API
        logger.debug(f"[{request_id}] Отправка запроса к API: {api_url}")
        logger.debug(f"[{request_id}] Полезная нагрузка: {json.dumps(payload)[:500]}")
        
        response = api_request("POST", api_url, headers=headers, json=payload, timeout=timeout, stream=False)
        logger.debug(f"[{request_id}] Код ответа API: {response.status_code}")
        
        # Обрабатываем ответ
        if response.status_code == 200:
            api_response = response.json()
            image_urls = extract_image_urls(api_response, request_id)
            
            if not image_urls:
                with TASKS_LOCK:
                    TASKS[task_id]['status'] = 'error'
                    TASKS[task_id]['error'] = "Не удалось извлечь URL изображений из ответа API"
                logger.error(f"[{request_id}] Не удалось извлечь URL изображений в задаче {task_id}")
                return
            
            logger.debug(f"[{request_id}] Успешно сгенерировано {len(image_urls)} изображений для задачи {task_id}")
            
            # Сохраняем параметры генерации для Midjourney
            if model in ["midjourney", "midjourney_6_1"]:
                for url in image_urls:
                    if url:
                        image_id_match = re.search(r'images/(\d+_\d+_\d+_\d+_\d+_\d+|\w+\d+)\.png', url)
                        if image_id_match:
                            image_id = image_id_match.group(1)
                            logger.info(f"[{request_id}] Извлечен image_id из URL: {image_id}")
                            gen_params = {
                                "mode": payload["promptObject"].get("mode", "fast"),
                                "aspect_width": payload["promptObject"].get("aspect_width", 1),
                                "aspect_height": payload["promptObject"].get("aspect_height", 1),
                                "isNiji6": payload["promptObject"].get("isNiji6", False),
                                "maintainModeration": payload["promptObject"].get("maintainModeration", True)
                            }
                            gen_params_key = f"gen_params:{image_id}"
                            safe_memcached_operation('set', gen_params_key, gen_params, expiry=3600*24*7)
                            logger.info(f"[{request_id}] Сохранены параметры генерации для {image_id}")
            
            full_image_urls = [get_full_url(url) for url in image_urls if url]
            
            # Формируем данные в формате OpenAI
            openai_data = []
            for i, url in enumerate(full_image_urls):
                if model in IMAGE_VARIATION_MODELS:
                    openai_data.append({
                        "url": url,
                        "revised_prompt": prompt,
                        "variation_commands": {"variation": f"/v{i + 1} {url}"}
                    })
                else:
                    openai_data.append({"url": url, "revised_prompt": prompt})
            
            # Создаем markdown для вывода
            markdown_text = ""
            if len(full_image_urls) == 1:
                markdown_text = f"![Image]({full_image_urls[0]}) `[_V1_]`"
            else:
                markdown_text = "\n".join([f"![Image {i+1}]({url}) `[_V{i+1}_]`" for i, url in enumerate(full_image_urls)])
            markdown_text += "\n\n> To generate **variants** of an **image** - tap (copy) **[_V1_]**" \
                             " - **[_V4_]** and send it (paste) in the next **prompt**"
            
            # Обновляем информацию о задаче
            with TASKS_LOCK:
                TASKS[task_id]['status'] = 'completed'
                TASKS[task_id]['result'] = {
                    "created": int(time.time()),
                    "data": openai_data,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": markdown_text,
                            "structured_output": {"type": "image", "image_urls": full_image_urls}
                        },
                        "index": 0,
                        "finish_reason": "stop"
                    }]
                }
            
            logger.info(f"[{request_id}] Задача {task_id} успешно выполнена")
        else:
            error_msg = "Неизвестная ошибка"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = error_data["error"]
            except:
                error_msg = f"HTTP ошибка: {response.status_code}"
            
            with TASKS_LOCK:
                TASKS[task_id]['status'] = 'error'
                TASKS[task_id]['error'] = error_msg
            
            logger.error(f"[{request_id}] Ошибка в задаче {task_id}: {error_msg}")
    except Exception as e:
        with TASKS_LOCK:
            TASKS[task_id]['status'] = 'error'
            TASKS[task_id]['error'] = str(e)
        logger.error(f"[{request_id}] Ошибка при выполнении асинхронной задачи {task_id}: {str(e)}")
    finally:
        # Закрываем сессию
        if 'session' in locals():
            session.close()

def create_async_task(api_key, payload, model, prompt, request_id):
    """
    Создает новую асинхронную задачу для генерации изображения
    
    Args:
        api_key: API ключ пользователя
        payload: Полезная нагрузка для API запроса
        model: Название модели
        prompt: Текст подсказки
        request_id: Идентификатор запроса
        
    Returns:
        str: Идентификатор созданной задачи
    """
    task_id = str(uuid.uuid4())
    
    # Сохраняем информацию о задаче
    with TASKS_LOCK:
        TASKS[task_id] = {
            'status': 'pending',
            'created_at': datetime.now(),
            'api_key': api_key,
            'model': model,
            'prompt': prompt,
        }
    
    # Запускаем асинхронную обработку в отдельном потоке
    thread = threading.Thread(
        target=process_image_generation_async,
        args=(task_id, api_key, payload, model, prompt, request_id)
    )
    thread.daemon = True
    thread.start()
    
    logger.info(f"[{request_id}] Создана асинхронная задача {task_id} для модели {model}")
    return task_id

def get_task_info(task_id):
    """
    Получает информацию о статусе задачи
    
    Args:
        task_id: Идентификатор задачи
        
    Returns:
        dict: Информация о задаче или None, если задача не найдена
    """
    with TASKS_LOCK:
        return TASKS.get(task_id) 
