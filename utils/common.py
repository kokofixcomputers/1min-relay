# utils/memcached.py
# Функции для работы с Memcached
import logging
import uuid
import threading
import json

# Получаем логгер, который уже инициализирован в app.py
logger = logging.getLogger("1min-relay")

def check_memcached_connection():
    """
    Checks the availability of Memcache, first in DoCker, then locally

    Returns:
        Tuple: (Bool, Str) - (Is Memcache available, connection line or none)
    """
    # I import Client here to avoid Name 'Client' Is Not Defined error
    try:
        from pymemcache.client.base import Client
    except ImportError:
        try:
            from memcache import Client
        except ImportError:
            logger.error("Failed to import Client from pymemcache or memcache")
            return False, None
            
    # Check Docker Memcache
    try:
        client = Client(("memcached", 11211))
        client.set("test_key", "test_value")
        if client.get("test_key") == b"test_value":
            client.delete("test_key")  # Clean up
            logger.info("Using memcached in Docker container")
            return True, "memcached://memcached:11211"
    except Exception as e:
        logger.debug(f"Docker memcached not available: {str(e)}")

    # Check the local Memcache
    try:
        client = Client(("127.0.0.1", 11211))
        client.set("test_key", "test_value")
        if client.get("test_key") == b"test_value":
            client.delete("test_key")  # Clean up
            logger.info("Using local memcached at 127.0.0.1:11211")
            return True, "memcached://127.0.0.1:11211"
    except Exception as e:
        logger.debug(f"Local memcached not available: {str(e)}")

    # If Memcache is not available
    logger.warning(
        "Memcached is not available. Using in-memory storage for rate limiting. Not-Recommended"
    )
    return False, None


logger.info(
    """
  _ __  __ _      ___     _           
 / |  \/  (_)_ _ | _ \___| |__ _ _  _ 
 | | |\/| | | ' \|   / -_) / _` | || |
 |_|_|  |_|_|_||_|_|_\___|_\__,_|\_, |
                                 |__/ """
)

# Closter function for safe access to Memcache
def safe_memcached_operation(operation, key, value=None, expiry=3600):
    """
    Safely performs operations on memcached, handling any exceptions.
    
    Args:
        operation (str): The operation to perform ('get', 'set', or 'delete')
        key (str): The key to operate on
        value (any, optional): The value to set (only for 'set' operation)
        expiry (int, optional): Expiry time in seconds (only for 'set' operation)
    
    Returns:
        The result of the operation or None if it failed
    """
    if MEMCACHED_CLIENT is None:
        # If Memcache is not available, we use the local storage
        if operation == 'get':
            return MEMORY_STORAGE.get(key, None)
        elif operation == 'set':
            MEMORY_STORAGE[key] = value
            logger.info(f"Saved in MEMORY_STORAGE: key={key}")
            return True
        elif operation == 'delete':
            if key in MEMORY_STORAGE:
                del MEMORY_STORAGE[key]
                return True
            return False
        return None
    
    try:
        if operation == 'get':
            result = MEMCACHED_CLIENT.get(key)
            if isinstance(result, bytes):
                try:
                    return json.loads(result.decode('utf-8'))
                except:
                    return result.decode('utf-8')
            return result
        elif operation == 'set':
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return MEMCACHED_CLIENT.set(key, value, time=expiry)
        elif operation == 'delete':
            return MEMCACHED_CLIENT.delete(key)
    except Exception as e:
        logger.error(f"Error in memcached operation {operation} on key {key}: {str(e)}")
        # When error Memcated, we also use a local storage
        if operation == 'get':
            return MEMORY_STORAGE.get(key, None)
        elif operation == 'set':
            MEMORY_STORAGE[key] = value
            logger.info(f"Saved in MEMORY_STORAGE due to memcached error: key={key}")
            return True
        elif operation == 'delete':
            if key in MEMORY_STORAGE:
                del MEMORY_STORAGE[key]
                return True
            return False
        return None
        
def delete_all_files_task():
    """
    Function for periodic deleting all user files
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting scheduled files cleanup task")

    try:
        # We get all users with files from MemcacheD
        if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
            # We get all the keys that begin with "user:"
            try:
                keys = []

                # Instead of scanning Slabs, we use a list of famous users
                # which should be saved when uploading files
                known_users = safe_memcached_operation('get', 'known_users_list')
                if known_users:
                    try:
                        if isinstance(known_users, str):
                            user_list = json.loads(known_users)
                        elif isinstance(known_users, bytes):
                            user_list = json.loads(known_users.decode('utf-8'))
                        else:
                            user_list = known_users

                        for user in user_list:
                            user_key = f"user:{user}" if not user.startswith("user:") else user
                            if user_key not in keys:
                                keys.append(user_key)
                    except Exception as e:
                        logger.warning(f"[{request_id}] Failed to parse known users list: {str(e)}")

                logger.info(f"[{request_id}] Found {len(keys)} user keys for cleanup")

                # We delete files for each user
                for user_key in keys:
                    try:
                        api_key = user_key.replace("user:", "")
                        user_files_json = safe_memcached_operation('get', user_key)

                        if not user_files_json:
                            continue

                        user_files = []
                        try:
                            if isinstance(user_files_json, str):
                                user_files = json.loads(user_files_json)
                            elif isinstance(user_files_json, bytes):
                                user_files = json.loads(user_files_json.decode('utf-8'))
                            else:
                                user_files = user_files_json
                        except:
                            continue

                        logger.info(f"[{request_id}] Cleaning up {len(user_files)} files for user {api_key[:8]}...")

                        # We delete each file
                        for file_info in user_files:
                            file_id = file_info.get("id")
                            if file_id:
                                try:
                                    delete_url = f"{ONE_MIN_ASSET_URL}/{file_id}"
                                    headers = {"API-KEY": api_key}

                                    delete_response = api_request("DELETE", delete_url, headers=headers)

                                    if delete_response.status_code == 200:
                                        logger.info(f"[{request_id}] Scheduled cleanup: deleted file {file_id}")
                                    else:
                                        logger.error(
                                            f"[{request_id}] Scheduled cleanup: failed to delete file {file_id}: {delete_response.status_code}")
                                except Exception as e:
                                    logger.error(
                                        f"[{request_id}] Scheduled cleanup: error deleting file {file_id}: {str(e)}")

                        # Cleaning the list of user files
                        safe_memcached_operation('set', user_key, json.dumps([]))
                        logger.info(f"[{request_id}] Cleared files list for user {api_key[:8]}")
                    except Exception as e:
                        logger.error(f"[{request_id}] Error processing user {user_key}: {str(e)}")
            except Exception as e:
                logger.error(f"[{request_id}] Error getting keys from memcached: {str(e)}")
    except Exception as e:
        logger.error(f"[{request_id}] Error in scheduled cleanup task: {str(e)}")

    # Plan the following execution in an hour
    cleanup_timer = threading.Timer(3600, delete_all_files_task)
    cleanup_timer.daemon = True
    cleanup_timer.start()
    logger.info(f"[{request_id}] Scheduled next cleanup in 1 hour")
