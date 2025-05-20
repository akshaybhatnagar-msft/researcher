# storage_client.py
from azure.data.tables import TableServiceClient, TableClient
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
import json
import os
from datetime import datetime

class TaskStorageClient:
    """Client for interacting with Azure Table Storage for task persistence"""
    
    def __init__(self, table_name="DeepResearchTasks"):
        """
        Initialize the storage client
        
        Args:
            table_name: Name of the table to use for storing tasks
        """
        self.table_name = table_name
        
        # Use DefaultAzureCredential if no connection string is provided
        try:
            try:
                credential = ManagedIdentityCredential()
                # Test if credential works
                credential.get_token("https://management.azure.com/.default")
            except Exception as e:
                # Fall back to DefaultAzureCredential if Managed Identity isn't available
                credential = DefaultAzureCredential()
                print(f"Warning: Using DefaultAzureCredential due to: {str(e)}")

            self.credential = credential
            self.endpoint = os.environ.get("AZURE_STORAGE_TABLE_URL")
            
            print(f"Initializing Table Storage client with DefaultAzureCredential at endpoint: {self.endpoint}")
            
            self.table_service = TableServiceClient(
                endpoint=self.endpoint,
                credential=self.credential
            )
        except Exception as e:
            print(f"Error initializing with DefaultAzureCredential: {e}")
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
                print(f"Creating table {self.table_name}...")
                self.table_service.create_table_if_not_exists(self.table_name)
                print(f"Table {self.table_name} created successfully")
            else:
                print(f"Table {self.table_name} already exists")
                
        except Exception as e:
            print(f"Error creating or checking table: {e}")
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
            print(f"Error getting table client: {e}")
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
        
        try:
            # Get table client and create entity
            table_client = self._get_table_client()
            table_client.create_entity(entity)
            task_id = entity["RowKey"]
            print(f"Created task {task_id} in Table Storage")
            return task_id
        except Exception as e:
            print(f"Error creating task in Table Storage: {e}")
            raise
    
    def get_task(self, task_id):
        """
        Get a task by ID
        
        Args:
            task_id: ID of the task to retrieve
            
        Returns:
            Dictionary containing task data or None if not found
        """
        table_client = self._get_table_client()
        
        try:
            entity = table_client.get_entity("task", task_id)
            return self._deserialize_json_fields(entity)
        except Exception:
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
        table_client = self._get_table_client()
        
        try:
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
        except Exception as e:
            print(f"Error updating task: {e}")
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
        table_client = self._get_table_client()
        
        try:
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
        except Exception as e:
            print(f"Error adding thought: {e}")
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
        table_client = self._get_table_client()
        
        try:
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
        except Exception as e:
            print(f"Error adding tool calls: {e}")
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
        return self.update_task(task_id, {"messages": messages})
    
    def list_tasks(self, created_by, max_results=100):
        """
        List all tasks
        
        Args:
            max_results: Maximum number of results to return
            
        Returns:
            List of task dictionaries
        """
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
    
    def delete_task(self, task_id):
        """
        Delete a task
        
        Args:
            task_id: ID of the task to delete
            
        Returns:
            Boolean indicating success or failure
        """
        table_client = self._get_table_client()
        
        try:
            table_client.delete_entity("task", task_id)
            return True
        except Exception as e:
            print(f"Error deleting task: {e}")
            return False