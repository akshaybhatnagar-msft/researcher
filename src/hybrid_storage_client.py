from azure.data.tables import TableServiceClient, TableClient
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.core.exceptions import AzureError
import json
import os
import time
import random
from datetime import datetime
from logging import getLogger

logger = getLogger(__name__)

class HybridTaskStorageClient:
    """Client for interacting with Azure Table Storage and Blob Storage for task persistence"""
    
    def __init__(self, table_name="DeepResearchTasks", container_name="task-data", max_retries=3, base_delay=1.0, max_delay=10.0):
        """
        Initialize the storage client
        
        Args:
            table_name: Name of the table to use for storing task metadata
            container_name: Name of the blob container to use for storing large task data
            max_retries: Maximum number of retries for operations (default: 3)
            base_delay: Base delay for exponential backoff in seconds (default: 1.0)
            max_delay: Maximum delay between retries in seconds (default: 10.0)
        """
        self.table_name = table_name
        self.container_name = container_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Use DefaultAzureCredential if no connection string is provided
        try:
            isLocal = os.environ.get("IS_LOCAL", "true").lower() == "true"
            try:
                if not isLocal:
                    # Only use ManagedIdentityCredential in non-local environments
                    credential = ManagedIdentityCredential()
                    # Test if credential works
                    credential.get_token("https://management.azure.com/.default")
                    print("Using ManagedIdentityCredential")
                else:
                    # Use DefaultAzureCredential in local environment
                    credential = DefaultAzureCredential()
                    print("Using DefaultAzureCredential for local environment")
            except Exception as e:
                # Fall back to DefaultAzureCredential if Managed Identity isn't available
                credential = DefaultAzureCredential()
                print(f"Warning: Using DefaultAzureCredential due to: {str(e)}")

            self.credential = credential
            self.table_endpoint = os.environ.get("AZURE_STORAGE_TABLE_URL")
            self.blob_endpoint = os.environ.get("AZURE_STORAGE_BLOB_URL")
            
            if not self.blob_endpoint:
                # If blob endpoint not specified, try to derive it from table endpoint
                if self.table_endpoint:
                    # Replace "table" with "blob" in the endpoint
                    self.blob_endpoint = self.table_endpoint.replace("table", "blob")
                    logger.info(f"Derived blob endpoint: {self.blob_endpoint}")
                else:
                    raise ValueError("Either AZURE_STORAGE_BLOB_URL or AZURE_STORAGE_TABLE_URL must be provided")
            
            logger.info(f"Initializing Table Storage client at endpoint: {self.table_endpoint}")
            logger.info(f"Initializing Blob Storage client at endpoint: {self.blob_endpoint}")
            
            self.table_service = TableServiceClient(
                endpoint=self.table_endpoint,
                credential=self.credential
            )
            
            self.blob_service = BlobServiceClient(
                account_url=self.blob_endpoint,
                credential=self.credential
            )
        except Exception as e:
            logger.error(f"Error initializing with Azure credentials: {e}", exc_info=True)
            raise
        
        # Create the table and container if they don't exist
        self._create_table_if_not_exists()
        self._create_container_if_not_exists()
    
    def _create_table_if_not_exists(self):
        """Create the tasks table if it doesn't exist"""
        try:
            # Check if the table exists by listing tables
            tables = list(self.table_service.list_tables())
            table_exists = any(table.name == self.table_name for table in tables)
            
            # If table doesn't exist, create it
            if not table_exists:
                logger.info(f"Creating table {self.table_name}...")
                self.table_service.create_table_if_not_exists(self.table_name)
                logger.info(f"Table {self.table_name} created successfully")
            else:
                logger.debug(f"Table {self.table_name} already exists")
                
        except Exception as e:
            logger.warning(f"Error creating or checking table: {e}", exc_info=True)
            # Continue execution even if table creation fails
            # The table might already exist or might be created by another process
    
    def _create_container_if_not_exists(self):
        """Create the blob container if it doesn't exist"""
        try:
            # Check if the container exists
            containers = list(self.blob_service.list_containers())
            container_exists = any(container.name == self.container_name for container in containers)
            
            # If container doesn't exist, create it
            if not container_exists:
                logger.info(f"Creating blob container {self.container_name}...")
                self.blob_service.create_container(self.container_name)
                logger.info(f"Blob container {self.container_name} created successfully")
            else:
                logger.debug(f"Blob container {self.container_name} already exists")
                
        except Exception as e:
            logger.warning(f"Error creating or checking blob container: {e}", exc_info=True)
    
    def _get_table_client(self):
        """Get a table client for the tasks table"""
        try:
            if self.table_service:
                return self.table_service.get_table_client(self.table_name)
            else:
                return TableClient(
                    endpoint=self.table_endpoint,
                    table_name=self.table_name,
                    credential=self.credential
                )
        except Exception as e:
            logger.error(f"Error getting table client: {e}", exc_info=True)
            raise
    
    def _get_blob_client(self, blob_name):
        """Get a blob client for the specified blob"""
        try:
            return self.blob_service.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
        except Exception as e:
            logger.error(f"Error getting blob client: {e}", exc_info=True)
            raise
    
    def _retry_operation(self, operation_name, operation_func, *args, **kwargs):
        """
        Execute an operation with retry logic using exponential backoff
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the operation function
            
        Returns:
            Result of the operation or raises the last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Attempting {operation_name} (attempt {attempt}/{self.max_retries})")
                result = operation_func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"{operation_name} succeeded after {attempt} attempts")
                return result
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    # Calculate backoff delay with jitter
                    delay = min(self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5), self.max_delay)
                    logger.warning(f"{operation_name} failed (attempt {attempt}/{self.max_retries}): {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"{operation_name} failed after {self.max_retries} attempts: {e}", exc_info=True)
        
        # If we reached here, all retries failed
        raise last_exception
    
    def _store_data_in_blob(self, task_id, field_name, data):
        """
        Store data in a blob
        
        Args:
            task_id: ID of the task
            field_name: Name of the field (used in blob name)
            data: Data to store (dict/list will be JSON serialized, other types stored as string)
            
        Returns:
            Blob name
        """
        blob_name = f"{task_id}/{field_name}.json"
        blob_client = self._get_blob_client(blob_name)
        
        # Convert data to JSON string if it's a dict or list
        if isinstance(data, (dict, list)):
            content = json.dumps(data)
        else:
            content = str(data)
        
        def _upload_blob():
            blob_client.upload_blob(content, overwrite=True)
            return blob_name
        
        return self._retry_operation(f"Upload blob {blob_name}", _upload_blob)
    
    def _get_data_from_blob(self, blob_name):
        """
        Get data from a blob
        
        Args:
            blob_name: Name of the blob to retrieve
            
        Returns:
            Data from the blob (deserialized from JSON if possible)
        """
        blob_client = self._get_blob_client(blob_name)
        
        def _download_blob():
            download_stream = blob_client.download_blob()
            content = download_stream.readall().decode('utf-8')
            
            # Try to deserialize from JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Not valid JSON, return as string
                return content
        
        return self._retry_operation(f"Download blob {blob_name}", _download_blob)
    
    def create_task(self, task_data):
        """
        Create a new task
        
        Args:
            task_data: Dictionary containing task data
            
        Returns:
            The task_id of the created task
        """
        # Create entity for Table Storage (metadata only)
        task_id = task_data['task_id']
        entity = {
            "PartitionKey": "task",
            "created_by": task_data.get('created_by', 'system'),
            "RowKey": task_id,
            "status": task_data['status'],
            "progress": task_data['progress'],
            "created_at": task_data['created_at'],
            "updated_at": task_data['updated_at'],
        }
        
        # Add error field if present (this is usually small)
        if "error" in task_data:
            entity["error"] = task_data["error"]
        
        # Store large fields in blobs
        large_fields = {
            "messages": task_data.get("messages", []),
            "thought_process": task_data.get("thought_process", []),
            "tool_calls": task_data.get("tool_calls", []),
        }
        
        # Store result in blob if present
        if "result" in task_data:
            large_fields["result"] = task_data["result"]
        
        # Add blob references to the entity
        blob_references = {}
        
        logger.info(f"Creating task {task_id}")
        
        def _create_task():
            # First store large fields in blobs
            for field_name, field_data in large_fields.items():
                if field_data:  # Only store non-empty data
                    blob_name = self._store_data_in_blob(task_id, field_name, field_data)
                    blob_references[field_name] = blob_name
            
            # Add blob references to entity
            entity["blob_references"] = json.dumps(blob_references)
            
            # Create entity in table
            table_client = self._get_table_client()
            table_client.create_entity(entity)
            
            return task_id
        
        try:
            return self._retry_operation(f"Create task {task_id}", _create_task)
        except Exception as e:
            logger.error(f"Error creating task {task_id} after retries: {e}", exc_info=True)
            raise
    
    def get_task(self, task_id):
        """
        Get a task by ID
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            Dictionary containing task data or None if not found
        """
        logger.debug(f"Getting task {task_id}")
        
        def _get_task():
            # Get metadata from table
            table_client = self._get_table_client()
            entity = table_client.get_entity("task", task_id)
            
            # Create task data dictionary
            task_data = dict(entity)
            
            # Load blob references
            blob_references = json.loads(entity.get("blob_references", "{}"))
            
            # Load data from blobs
            for field_name, blob_name in blob_references.items():
                task_data[field_name] = self._get_data_from_blob(blob_name)
            
            # Remove blob_references field from result
            if "blob_references" in task_data:
                del task_data["blob_references"]
            
            return task_data
        
        try:
            return self._retry_operation(f"Get task {task_id}", _get_task)
        except Exception as e:
            logger.warning(f"Task {task_id} not found or error retrieving: {e}")
            return None
    
    def update_task(self, task_id, updates):
        """
        Update a task
        
        Args:
            task_id: ID of the task to update
            updates: Dictionary containing fields to update
            
        Returns:
            Boolean indicating success or failure
        """
        logger.info(f"Updating task {task_id} with fields: {list(updates.keys())}")
        
        def _update_task():
            table_client = self._get_table_client()
            
            # Get current entity
            entity = table_client.get_entity("task", task_id)
            
            # Update timestamp
            entity["updated_at"] = datetime.now().isoformat()
            
            # Load blob references
            blob_references = json.loads(entity.get("blob_references", "{}"))
            
            # Process updates
            table_updates = {}
            blob_updates = {}
            
            for key, value in updates.items():
                # Check if the field should be stored in a blob
                if key in ["messages", "thought_process", "tool_calls", "result"] or isinstance(value, (dict, list)) and len(json.dumps(value)) > 30000:
                    # Store in blob
                    blob_updates[key] = value
                else:
                    # Store in table
                    table_updates[key] = value
            
            # Update blobs
            for field_name, field_data in blob_updates.items():
                blob_name = self._store_data_in_blob(task_id, field_name, field_data)
                blob_references[field_name] = blob_name
            
            # Update entity with table updates and blob references
            for key, value in table_updates.items():
                entity[key] = value
            
            entity["blob_references"] = json.dumps(blob_references)
            
            # Update entity in table
            table_client.update_entity(entity)
            return True
        
        try:
            return self._retry_operation(f"Update task {task_id}", _update_task)
        except Exception as e:
            logger.error(f"Error updating task {task_id} after retries: {e}", exc_info=True)
            return False
    
    def update_task_status(self, task_id, status, progress=None, result=None, error=None):
        """
        Update a task's status
        
        Args:
            task_id: ID of the task to update
            status: New status value
            progress: New progress value (optional)
            result: New result value (optional)
            error: Error message (optional)
            
        Returns:
            Boolean indicating success or failure
        """
        updates = {"status": status}
        
        if progress is not None:
            updates["progress"] = progress
        
        if result is not None:
            updates["result"] = result
        
        if error is not None:
            updates["error"] = error
        
        logger.info(f"Updating task {task_id} status to '{status}'")
        return self.update_task(task_id, updates)
    
    def add_thought_to_task(self, task_id, thought):
        """
        Add a thought to a task's thought process
        
        Args:
            task_id: ID of the task to update
            thought: Thought to add
            
        Returns:
            Boolean indicating success or failure
        """
        logger.debug(f"Adding thought to task {task_id}")
        
        def _add_thought():
            # Get task data
            task_data = self.get_task(task_id)
            if not task_data:
                raise ValueError(f"Task {task_id} not found")
            
            # Update timestamp
            timestamp = datetime.now().isoformat()
            
            # Get current thought process
            thought_process = task_data.get("thought_process", [])
            
            # Add new thought
            thought_process.append({
                "timestamp": timestamp,
                "content": thought
            })
            
            # Update task
            return self.update_task(task_id, {
                "thought_process": thought_process,
                "updated_at": timestamp
            })
        
        try:
            return self._retry_operation(f"Add thought to task {task_id}", _add_thought)
        except Exception as e:
            logger.error(f"Error adding thought to task {task_id} after retries: {e}", exc_info=True)
            return False
    
    def add_tool_call_to_task(self, task_id, tool_call_data):
        """
        Add tool calls to a task
        
        Args:
            task_id: ID of the task to update
            tool_call_data: Tool call data to add
            
        Returns:
            Boolean indicating success or failure
        """
        logger.debug(f"Adding tool calls to task {task_id}")
        
        def _add_tool_calls():
            # Get task data
            task_data = self.get_task(task_id)
            if not task_data:
                raise ValueError(f"Task {task_id} not found")
            
            # Update timestamp
            timestamp = datetime.now().isoformat()
            
            # Get current tool calls
            tool_calls = task_data.get("tool_calls", [])
            
            # Build a map of existing tool calls by ID
            existing_calls = {
                call["data"][0]["id"]: call
                for call in tool_calls
                if call.get("data", [])
            }
            
            # Now merge/overwrite with the new ones
            for call in tool_call_data:
                call_id = call["id"]
                existing_calls[call_id] = {
                    "timestamp": timestamp,
                    "data": [call]
                }
            
            # Replace the tool_calls with deduplicated values
            new_tool_calls = list(existing_calls.values())
            
            # Update task
            return self.update_task(task_id, {
                "tool_calls": new_tool_calls,
                "updated_at": timestamp
            })
        
        try:
            return self._retry_operation(f"Add tool calls to task {task_id}", _add_tool_calls)
        except Exception as e:
            logger.error(f"Error adding tool calls to task {task_id} after retries: {e}", exc_info=True)
            return False
    
    def update_task_messages(self, task_id, messages):
        """
        Update a task's messages
        
        Args:
            task_id: ID of the task to update
            messages: New messages
            
        Returns:
            Boolean indicating success or failure
        """
        logger.info(f"Updating messages for task {task_id}")
        return self.update_task(task_id, {"messages": messages})
    
    def list_tasks(self, created_by, max_results=100):
        """
        List all tasks
        
        Args:
            created_by: User or system that created the tasks
            max_results: Maximum number of results to return
            
        Returns:
            List of task dictionaries (metadata only without blob data)
        """
        logger.info(f"Listing tasks created by '{created_by}' (max: {max_results})")
        
        def _list_tasks():
            table_client = self._get_table_client()
            
            # Query tasks
            query_filter = f"PartitionKey eq 'task' and created_by eq '{created_by}'"
            entities = table_client.query_entities(query_filter, results_per_page=max_results)
            
            # Process entities
            tasks = []
            for entity in entities:
                task_data = dict(entity)
                
                # Remove blob_references field
                if "blob_references" in task_data:
                    del task_data["blob_references"]
                
                tasks.append(task_data)
            
            return tasks
        
        try:
            return self._retry_operation(f"List tasks by '{created_by}'", _list_tasks)
        except Exception as e:
            logger.error(f"Error listing tasks after retries: {e}", exc_info=True)
            return []
    
    def delete_task(self, task_id):
        """
        Delete a task and its associated blobs
        
        Args:
            task_id: ID of the task to delete
            
        Returns:
            Boolean indicating success or failure
        """
        logger.info(f"Deleting task {task_id}")
        
        def _delete_task():
            # Get task metadata to find blob references
            table_client = self._get_table_client()
            entity = table_client.get_entity("task", task_id)
            
            # Delete blobs if present
            if "blob_references" in entity:
                blob_references = json.loads(entity["blob_references"])
                for blob_name in blob_references.values():
                    blob_client = self._get_blob_client(blob_name)
                    try:
                        blob_client.delete_blob()
                    except Exception as e:
                        logger.warning(f"Error deleting blob {blob_name}: {e}")
            
            # Delete task from table
            table_client.delete_entity("task", task_id)
            return True
        
        try:
            return self._retry_operation(f"Delete task {task_id}", _delete_task)
        except Exception as e:
            logger.error(f"Error deleting task {task_id} after retries: {e}", exc_info=True)
            return False