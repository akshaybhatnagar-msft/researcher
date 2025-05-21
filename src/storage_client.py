from azure.data.tables import TableServiceClient, TableClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.core.exceptions import AzureError
import json
import os
import time
import random
from datetime import datetime
from logging import getLogger

logger = getLogger(__name__)

class TaskStorageClient:
    """Client for interacting with Azure Table Storage for task persistence"""
    
    def __init__(self, table_name="DeepResearchTasks", max_retries=3, base_delay=1.0, max_delay=10.0):
        """
        Initialize the storage client
        
        Args:
            table_name: Name of the table to use for storing tasks
            max_retries: Maximum number of retries for operations (default: 3)
            base_delay: Base delay for exponential backoff in seconds (default: 1.0)
            max_delay: Maximum delay between retries in seconds (default: 10.0)
        """
        self.table_name = table_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Use DefaultAzureCredential if no connection string is provided
        try:
            isLocal = os.environ.get("IS_LOCAL", "false").lower() == "true"
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
            self.endpoint = os.environ.get("AZURE_STORAGE_TABLE_URL")
            
            logger.info(f"Initializing Table Storage client at endpoint: {self.endpoint}")
            
            self.table_service = TableServiceClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
        except Exception as e:
            logger.error(f"Error initializing with Azure credentials: {e}", exc_info=True)
            raise
        
        # Create the table if it doesn't exist
        self._create_table_if_not_exists()
    
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
    
    def _get_table_client(self):
        """Get a table client for the tasks table"""
        try:
            if self.table_service:
                return self.table_service.get_table_client(self.table_name)
            else:
                return TableClient(
                    endpoint=self.endpoint,
                    table_name=self.table_name,
                    credential=self.credential
                )
        except Exception as e:
            logger.error(f"Error getting table client: {e}", exc_info=True)
            raise
    
    def _serialize_json_fields(self, entity):
        """Serialize JSON fields for storage"""
        serialized = dict(entity)
        for key, value in serialized.items():
            if isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value)
        return serialized
    
    def _deserialize_json_fields(self, entity):
        """Deserialize JSON fields from storage"""
        deserialized = dict(entity)
        for key, value in deserialized.items():
            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                try:
                    deserialized[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Not valid JSON, keep as string
                    pass
        return deserialized
    
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
        
    def create_task(self, task_data):
        """
        Create a new task in Table Storage
        
        Args:
            task_data: Dictionary containing task data
            
        Returns:
            The task_id of the created task
        """
        
        # Create entity for Table Storage
        entity = {
            "PartitionKey": "task",
            "created_by": task_data.get('created_by', 'system'),
            "RowKey": task_data['task_id'],
            "status": task_data['status'],
            "progress": task_data['progress'],
            "messages": json.dumps(task_data.get("messages", [])),
            "thought_process": json.dumps([]),
            "tool_calls": json.dumps([]),
            "created_at": task_data['created_at'],
            "updated_at": task_data['updated_at'],
        }
        
        # Add other fields if present
        if "error" in task_data:
            entity["error"] = task_data["error"]
        
        if "result" in task_data:
            entity["result"] = json.dumps(task_data["result"])
        
        task_id = entity["RowKey"]
        logger.info(f"Creating task {task_id} in Table Storage")
        
        def _create_entity():
            table_client = self._get_table_client()
            table_client.create_entity(entity)
            return task_id
        
        try:
            return self._retry_operation(f"Create task {task_id}", _create_entity)
        except Exception as e:
            logger.error(f"Error creating task {task_id} in Table Storage after retries: {e}", exc_info=True)
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
        
        def _get_entity():
            table_client = self._get_table_client()
            entity = table_client.get_entity("task", task_id)
            return self._deserialize_json_fields(entity)
        
        try:
            return self._retry_operation(f"Get task {task_id}", _get_entity)
        except Exception as e:
            logger.warning(f"Task {task_id} not found or error retrieving: {e}")
            return None
    
    def update_task(self, task_id, updates):
        """
        Update a task in Table Storage
        
        Args:
            task_id: ID of the task to update
            updates: Dictionary containing fields to update
            
        Returns:
            Boolean indicating success or failure
        """
        logger.info(f"Updating task {task_id} with fields: {list(updates.keys())}")
        
        def _update_entity():
            table_client = self._get_table_client()
            
            # Get current entity
            entity = table_client.get_entity("task", task_id)
            
            # Update timestamp
            entity["updated_at"] = datetime.now().isoformat()
            
            # Update fields
            for key, value in updates.items():
                if isinstance(value, (dict, list)):
                    entity[key] = json.dumps(value)
                else:
                    entity[key] = value
            
            # Update entity in table
            table_client.update_entity(entity)
            return True
        
        try:
            return self._retry_operation(f"Update task {task_id}", _update_entity)
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
            table_client = self._get_table_client()
            
            # Get current entity
            entity = table_client.get_entity("task", task_id)
            
            # Update timestamp
            timestamp = datetime.now().isoformat()
            entity["updated_at"] = timestamp
            
            # Get current thought process
            thought_process = json.loads(entity.get("thought_process", "[]"))
            
            # Add new thought
            thought_process.append({
                "timestamp": timestamp,
                "content": thought
            })
            
            # Update entity
            entity["thought_process"] = json.dumps(thought_process)
            table_client.update_entity(entity)
            return True
            
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
            table_client = self._get_table_client()
            
            # Get current entity
            entity = table_client.get_entity("task", task_id)
            
            # Update timestamp
            timestamp = datetime.now().isoformat()
            entity["updated_at"] = timestamp
            
            # Get current tool calls
            tool_calls = json.loads(entity.get("tool_calls", "[]"))
            
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
            entity["tool_calls"] = json.dumps(list(existing_calls.values()))
            table_client.update_entity(entity)
            return True
            
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
            List of task dictionaries
        """
        logger.info(f"Listing tasks created by '{created_by}' (max: {max_results})")
        
        def _list_tasks():
            table_client = self._get_table_client()
            
            # Query tasks
            query_filter = f"PartitionKey eq 'task' and created_by eq '{created_by}'"
            entities = table_client.query_entities(query_filter, results_per_page=max_results)
            
            # Deserialize and return tasks
            tasks = []
            for entity in entities:
                deserialized = self._deserialize_json_fields(entity)
                tasks.append(deserialized)
            
            return tasks
            
        try:
            return self._retry_operation(f"List tasks by '{created_by}'", _list_tasks)
        except Exception as e:
            logger.error(f"Error listing tasks after retries: {e}", exc_info=True)
            return []
    
    def delete_task(self, task_id):
        """
        Delete a task
        
        Args:
            task_id: ID of the task to delete
            
        Returns:
            Boolean indicating success or failure
        """
        logger.info(f"Deleting task {task_id}")
        
        def _delete_task():
            table_client = self._get_table_client()
            table_client.delete_entity("task", task_id)
            return True
            
        try:
            return self._retry_operation(f"Delete task {task_id}", _delete_task)
        except Exception as e:
            logger.error(f"Error deleting task {task_id} after retries: {e}", exc_info=True)
            return False