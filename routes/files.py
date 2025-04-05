# Маршруты для работы с файлами
# Functions for working with files in API
@app.route("/v1/files", methods=["GET", "POST", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_files():
    """
    Route for working with files: getting a list and downloading new files
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Get - getting a list of files
    if request.method == "GET":
        logger.info(f"[{request_id}] Received request: GET /v1/files")
        try:
            # We get a list of user files from MemcacheD
            files = []
            if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
                try:
                    user_key = f"user:{api_key}"
                    user_files_json = safe_memcached_operation('get', user_key)

                    if user_files_json:
                        try:
                            if isinstance(user_files_json, str):
                                user_files = json.loads(user_files_json)
                            elif isinstance(user_files_json, bytes):
                                user_files = json.loads(user_files_json.decode('utf-8'))
                            else:
                                user_files = user_files_json

                            # Let's convert files about files to API response format
                            for file_info in user_files:
                                if isinstance(file_info, dict) and "id" in file_info:
                                    files.append({
                                        "id": file_info.get("id"),
                                        "object": "file",
                                        "bytes": file_info.get("bytes", 0),
                                        "created_at": file_info.get("created_at", int(time.time())),
                                        "filename": file_info.get("filename", f"file_{file_info.get('id')}"),
                                        "purpose": "assistants",
                                        "status": "processed"
                                    })
                            logger.debug(f"[{request_id}] Found {len(files)} files for user in memcached")
                        except Exception as e:
                            logger.error(f"[{request_id}] Error parsing user files from memcached: {str(e)}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")

            # We form an answer in Openai API format
            response_data = {
                "data": files,
                "object": "list"
            }
            response = make_response(jsonify(response_data))
            set_response_headers(response)
            return response
        except Exception as e:
            logger.error(f"[{request_id}] Exception during files list request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Post - downloading a new file
    elif request.method == "POST":
        logger.info(f"[{request_id}] Received request: POST /v1/files")

        # Checking a file
        if "file" not in request.files:
            logger.error(f"[{request_id}] No file provided")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        purpose = request.form.get("purpose", "assistants")

        try:
            # We get the contents of the file
            file_data = file.read()
            file_name = file.filename

            # We get a loaded file ID
            file_id = upload_document(file_data, file_name, api_key, request_id)

            if not file_id:
                logger.error(f"[{request_id}] Failed to upload file")
                return jsonify({"error": "Failed to upload file"}), 500

            # We form an answer in Openai API format
            response_data = {
                "id": file_id,
                "object": "file",
                "bytes": len(file_data),
                "created_at": int(time.time()),
                "filename": file_name,
                "purpose": purpose,
                "status": "processed"
            }

            logger.info(f"[{request_id}] File uploaded successfully: {file_id}")
            response = make_response(jsonify(response_data))
            set_response_headers(response)
            return response

        except Exception as e:
            logger.error(f"[{request_id}] Exception during file upload: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route("/v1/files/<file_id>", methods=["GET", "DELETE", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file(file_id):
    """
    Route for working with a specific file: obtaining information and deleting
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    # Get - obtaining file information
    if request.method == "GET":
        logger.info(f"[{request_id}] Received request: GET /v1/files/{file_id}")
        try:
            # We are looking for a file in saved user files in Memcache
            file_info = None
            if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
                try:
                    user_key = f"user:{api_key}"
                    user_files_json = safe_memcached_operation('get', user_key)

                    if user_files_json:
                        try:
                            if isinstance(user_files_json, str):
                                user_files = json.loads(user_files_json)
                            elif isinstance(user_files_json, bytes):
                                user_files = json.loads(user_files_json.decode('utf-8'))
                            else:
                                user_files = user_files_json

                            # Looking for a file with the specified ID
                            for file_item in user_files:
                                if file_item.get("id") == file_id:
                                    file_info = file_item
                                    logger.debug(f"[{request_id}] Found file info in memcached: {file_id}")
                                    break
                        except Exception as e:
                            logger.error(f"[{request_id}] Error parsing user files from memcached: {str(e)}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")

            # If the file is not found, we return the filler
            if not file_info:
                logger.debug(f"[{request_id}] File not found in memcached, using placeholder: {file_id}")
                file_info = {
                    "id": file_id,
                    "bytes": 0,
                    "created_at": int(time.time()),
                    "filename": f"file_{file_id}"
                }

            # We form an answer in Openai API format
            response_data = {
                "id": file_info.get("id"),
                "object": "file",
                "bytes": file_info.get("bytes", 0),
                "created_at": file_info.get("created_at", int(time.time())),
                "filename": file_info.get("filename", f"file_{file_id}"),
                "purpose": "assistants",
                "status": "processed"
            }

            response = make_response(jsonify(response_data))
            set_response_headers(response)
            return response

        except Exception as e:
            logger.error(f"[{request_id}] Exception during file info request: {str(e)}")
            return jsonify({"error": str(e)}), 500

    # Delete - File deletion
    elif request.method == "DELETE":
        logger.info(f"[{request_id}] Received request: DELETE /v1/files/{file_id}")
        try:
            # If the files are stored in Memcached, delete the file from the list
            deleted = False
            if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
                try:
                    user_key = f"user:{api_key}"
                    user_files_json = safe_memcached_operation('get', user_key)

                    if user_files_json:
                        try:
                            if isinstance(user_files_json, str):
                                user_files = json.loads(user_files_json)
                            elif isinstance(user_files_json, bytes):
                                user_files = json.loads(user_files_json.decode('utf-8'))
                            else:
                                user_files = user_files_json

                            # We filter the list, excluding the file with the specified ID
                            new_user_files = [f for f in user_files if f.get("id") != file_id]

                            # If the list has changed, we save the updated list
                            if len(new_user_files) < len(user_files):
                                safe_memcached_operation('set', user_key, json.dumps(new_user_files))
                                logger.info(f"[{request_id}] Deleted file {file_id} from user's files in memcached")
                                deleted = True
                        except Exception as e:
                            logger.error(f"[{request_id}] Error updating user files in memcached: {str(e)}")
                except Exception as e:
                    logger.error(f"[{request_id}] Error retrieving user files from memcached: {str(e)}")

            # Return the answer about successful removal (even if the file was not found)
            response_data = {
                "id": file_id,
                "object": "file",
                "deleted": True
            }

            response = make_response(jsonify(response_data))
            set_response_headers(response)
            return response

        except Exception as e:
            logger.error(f"[{request_id}] Exception during file deletion: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route("/v1/files/<file_id>/content", methods=["GET", "OPTIONS"])
@limiter.limit("60 per minute")
def handle_file_content(file_id):
    """
    Route for obtaining the contents of the file
    """
    if request.method == "OPTIONS":
        return handle_options_request()

    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received request: GET /v1/files/{file_id}/content")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    try:
        # In 1min.ai there is no API to obtain the contents of the file by ID
        # Return the error
        logger.error(f"[{request_id}] File content retrieval not supported")
        return jsonify({"error": "File content retrieval not supported"}), 501

    except Exception as e:
        logger.error(f"[{request_id}] Exception during file content request: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/files", methods=["POST"])
@limiter.limit("60 per minute")
def upload_file():
    """
    File download route (analogue Openai Files API)
    """
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Received file upload request")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.error(f"[{request_id}] Invalid Authentication")
        return ERROR_HANDLER(1021)

    api_key = auth_header.split(" ")[1]

    if "file" not in request.files:
        logger.error(f"[{request_id}] No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.error(f"[{request_id}] No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        # We save the file in memory
        file_data = file.read()
        file_name = file.filename

        # We download the file to the 1min.ai server
        file_id = upload_document(file_data, file_name, api_key, request_id)

        if not file_id:
            return jsonify({"error": "Failed to upload file"}), 500

        # We save the file of the file in the user's session through Memcache, if it is available
        if 'MEMCACHED_CLIENT' in globals() and MEMCACHED_CLIENT is not None:
            try:
                user_key = f"user:{api_key}"
                # We get the current user's current files or create a new list
                user_files_json = safe_memcached_operation('get', user_key)
                user_files = []

                if user_files_json:
                    try:
                        if isinstance(user_files_json, str):
                            user_files = json.loads(user_files_json)
                        elif isinstance(user_files_json, bytes):
                            user_files = json.loads(user_files_json.decode('utf-8'))
                    except Exception as e:
                        logger.error(f"[{request_id}] Error parsing user files from memcached: {str(e)}")
                        user_files = []

                # Add a new file
                file_info = {
                    "id": file_id,
                    "filename": file_name,
                    "uploaded_at": int(time.time())
                }

                # Check that a file with such an ID is not yet on the list
                if not any(f.get("id") == file_id for f in user_files):
                    user_files.append(file_info)

                # We save the updated file list
                safe_memcached_operation('set', user_key, json.dumps(user_files))
                logger.info(f"[{request_id}] Saved file ID {file_id} for user in memcached")

                # Add the user to the list of well -known users for cleaning function
                known_users_list_json = safe_memcached_operation('get', 'known_users_list')
                known_users_list = []

                if known_users_list_json:
                    try:
                        if isinstance(known_users_list_json, str):
                            known_users_list = json.loads(known_users_list_json)
                        elif isinstance(known_users_list_json, bytes):
                            known_users_list = json.loads(known_users_list_json.decode('utf-8'))
                    except Exception as e:
                        logger.error(f"[{request_id}] Error parsing known users list: {str(e)}")

                # Add the API key to the list of famous users if it is not yet
                if api_key not in known_users_list:
                    known_users_list.append(api_key)
                    safe_memcached_operation('set', 'known_users_list', json.dumps(known_users_list))
                    logger.debug(f"[{request_id}] Added user to known_users_list for cleanup")
            except Exception as e:
                logger.error(f"[{request_id}] Error saving file info to memcached: {str(e)}")

        # We create an answer in the Openai API format
        response_data = {
            "id": file_id,
            "object": "file",
            "bytes": len(file_data),
            "created_at": int(time.time()),
            "filename": file_name,
            "purpose": request.form.get("purpose", "assistants")
        }

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"[{request_id}] Exception during file upload: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Вспомогательные функции для работы с файлами
def upload_document(file_data, file_name, api_key, request_id=None):
    """
    Downloads the file/document to the server and returns its ID.

    Args:
        File_DATA: Binar file contents
        File_name: file name
        API_KEY: user API
        Request_id: ID Request for Logging

    Returns:
        STR: ID loaded file or None in case of error
    """
    session = create_session()
    try:
        # Determine the type of expansion file
        extension = os.path.splitext(file_name)[1].lower()
        logger.info(f"[{request_id}] Uploading document: {file_name}")

        # Dictionary with MIME types for different file extensions
        mime_types = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".csv": "text/csv",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".json": "application/json",
            ".md": "text/markdown",
            ".html": "text/html",
            ".htm": "text/html",
            ".xml": "application/xml",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
            ".rtf": "application/rtf",
        }

        # We get MIME-type from a dictionary or use Octet-Stream by default
        mime_type = mime_types.get(extension, "application/octet-stream")

        # Determine the type of file for special processing
        file_type = None
        if extension in [".doc"]:
            file_type = "DOC"
        elif extension in [".docx"]:
            file_type = "DOCX"

        # We download the file to the server - add more details to the logs
        logger.info(
            f"[{request_id}] Uploading file to 1min.ai: {file_name} ({mime_type}, {len(file_data)} bytes)"
        )

        headers = {"API-KEY": api_key}

        # Special headlines for DOC/DOCX
        if file_type in ["DOC", "DOCX"]:
            headers["X-File-Type"] = file_type

        files = {"asset": (file_name, file_data, mime_type)}

        upload_response = session.post(ONE_MIN_ASSET_URL, headers=headers, files=files)

        if upload_response.status_code != 200:
            logger.error(
                f"[{request_id}] Document upload failed: {upload_response.status_code} - {upload_response.text}"
            )
            return None

        # Detailed logistics of the answer
        try:
            response_text = upload_response.text
            logger.debug(
                f"[{request_id}] Raw upload response: {response_text[:500]}..."
            )

            response_data = upload_response.json()
            logger.debug(
                f"[{request_id}] Upload response JSON: {json.dumps(response_data)[:500]}..."
            )

            file_id = None
            if "id" in response_data:
                file_id = response_data["id"]
                logger.debug(f"[{request_id}] Found file ID at top level: {file_id}")
            elif (
                    "fileContent" in response_data and "id" in response_data["fileContent"]
            ):
                file_id = response_data["fileContent"]["id"]
                logger.debug(f"[{request_id}] Found file ID in fileContent: {file_id}")
            elif (
                    "fileContent" in response_data and "uuid" in response_data["fileContent"]
            ):
                file_id = response_data["fileContent"]["uuid"]
                logger.debug(f"[{request_id}] Found file ID (uuid) in fileContent: {file_id}")
            else:
                # We are trying to find ID in other places of response structure
                if isinstance(response_data, dict):
                    # Recursive search for ID in the response structure
                    def find_id(obj, path="root"):
                        if isinstance(obj, dict):
                            if "id" in obj:
                                logger.debug(
                                    f"[{request_id}] Found ID at path '{path}': {obj['id']}"
                                )
                                return obj["id"]
                            if "uuid" in obj:
                                logger.debug(
                                    f"[{request_id}] Found UUID at path '{path}': {obj['uuid']}"
                                )
                                return obj["uuid"]
                            for k, v in obj.items():
                                result = find_id(v, f"{path}.{k}")
                                if result:
                                    return result
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                result = find_id(item, f"{path}[{i}]")
                                if result:
                                    return result
                        return None

                    file_id = find_id(response_data)

            if not file_id:
                logger.error(
                    f"[{request_id}] Could not find file ID in response: {json.dumps(response_data)}"
                )
                return None

            logger.info(
                f"[{request_id}] Document uploaded successfully. File ID: {file_id}"
            )
            return file_id
        except Exception as e:
            logger.error(f"[{request_id}] Error parsing upload response: {str(e)}")
            traceback.print_exc()
            return None
    except Exception as e:
        logger.error(f"[{request_id}] Error uploading document: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        session.close()

def create_conversation_with_files(file_ids, title, model, api_key, request_id=None):
    """
    Creates a new conversation with files

    Args:
        File_ids: List of ID files
        Title: The name of the conversation
        Model: Model AI
        API_KEY: API Key
        Request_id: ID Request for Logging

    Returns:
        STR: ID conversations or None in case of error
    """
    request_id = request_id or str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Creating conversation with {len(file_ids)} files")

    try:
        # We form Payload for a request with the right field names
        payload = {
            "title": title,
            "type": "CHAT_WITH_PDF",
            "model": model,
            "fileIds": file_ids,  # Using the correct name of the field 'Fileds' instead of 'Filelist'
        }

        logger.debug(f"[{request_id}] Conversation payload: {json.dumps(payload)}")

        # We use the correct URL API C /API /
        conversation_url = "https://api.1min.ai/api/features/conversations?type=CHAT_WITH_PDF"

        logger.debug(f"[{request_id}] Creating conversation using URL: {conversation_url}")

        headers = {"API-KEY": api_key, "Content-Type": "application/json"}
        response = requests.post(conversation_url, json=payload, headers=headers)

        logger.debug(f"[{request_id}] Create conversation response status: {response.status_code}")

        if response.status_code != 200:
            logger.error(
                f"[{request_id}] Failed to create conversation: {response.status_code} - {response.text}"
            )
            return None

        response_data = response.json()
        logger.debug(f"[{request_id}] Conversation response data: {json.dumps(response_data)}")

        # Looking for ID conversations in different places of answer
        conversation_id = None
        if "conversation" in response_data and "uuid" in response_data["conversation"]:
            conversation_id = response_data["conversation"]["uuid"]
        elif "id" in response_data:
            conversation_id = response_data["id"]
        elif "uuid" in response_data:
            conversation_id = response_data["uuid"]

        # Recursive search for ID conversations in the structure of the response
        if not conversation_id:
            def find_conversation_id(obj, path=""):
                if isinstance(obj, dict):
                    if "id" in obj:
                        logger.debug(f"[{request_id}] Found ID at path '{path}.id': {obj['id']}")
                        return obj["id"]
                    if "uuid" in obj:
                        logger.debug(f"[{request_id}] Found UUID at path '{path}.uuid': {obj['uuid']}")
                        return obj["uuid"]

                    for key, value in obj.items():
                        result = find_conversation_id(value, f"{path}.{key}")
                        if result:
                            return result
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        result = find_conversation_id(item, f"{path}[{i}]")
                        if result:
                            return result
                return None

            conversation_id = find_conversation_id(response_data)

        if not conversation_id:
            logger.error(
                f"[{request_id}] Could not find conversation ID in response: {response_data}"
            )
            return None

        logger.info(
            f"[{request_id}] Conversation created successfully: {conversation_id}"
        )
        return conversation_id
    except Exception as e:
        logger.error(f"[{request_id}] Error creating conversation: {str(e)}")
        traceback.print_exc()
        return None
