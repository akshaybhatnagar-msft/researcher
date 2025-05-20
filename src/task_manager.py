# task_manager.py
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import HTTPException

def create_task(storage_client=None, created_by=None) -> str:
    """
    Create a new task and return its ID
    
    Args:
        storage_client: Optional storage client for persistence
        
    Returns:
        task_id: ID of the created task
    """
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    task_data = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "messages": [],
        "thought_process": [],
        "tool_calls": [],
        "created_at": timestamp,
        "updated_at": timestamp,
        "created_by": created_by,
    }
    
    if storage_client:
        # Create task in storage
        storage_client.create_task(task_data)
    else:
        raise ValueError("Storage client not initialized")
    
    return task_id

def update_task_status(task_id: str, status: str, progress: float = None, result: Dict[str, Any] = None, error: str = None, storage_client=None) -> None:
    """
    Update a task's status
    
    Args:
        task_id: ID of the task to update
        status: New status value
        progress: New progress value (optional)
        result: New result value (optional)
        error: Error message (optional)
        storage_client: Storage client for persistence
    """
    if not storage_client:
        raise ValueError("Storage client not initialized")
    
    # Prepare updates
    updates = {"status": status}
    
    if progress is not None:
        updates["progress"] = progress
    
    if result is not None:
        updates["result"] = result
    
    if error is not None:
        updates["error"] = error
    
    # Update in storage
    success = storage_client.update_task(task_id, updates)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or update failed")

def update_task_messages(task_id: str, messages: List[Dict[str, Any]], storage_client=None) -> None:
    """
    Update a task's messages
    
    Args:
        task_id: ID of the task to update
        messages: New messages
        storage_client: Storage client for persistence
    """
    if not storage_client:
        raise ValueError("Storage client not initialized")
    
    # Update in storage
    success = storage_client.update_task_messages(task_id, messages)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or update failed")

def get_task(task_id: str, storage_client=None, created_by=None):
    """
    Get a task by ID
    
    Args:
        task_id: ID of the task to retrieve
        storage_client: Storage client for persistence
        
    Returns:
        Task data or raises an HTTPException if not found
    """
    if not storage_client:
        raise ValueError("Storage client not initialized")
    
    task = storage_client.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    if not task.get("created_by") != created_by:
        raise HTTPException(status_code=403, detail=f"Task {task_id} not found")
    
    return task

def list_tasks(storage_client=None, created_by=None, max_results=100):
    """
    List all tasks
    
    Args:
        storage_client: Storage client for persistence
        max_results: Maximum number of results to return
        
    Returns:
        List of tasks
    """
    if not storage_client:
        raise ValueError("Storage client not initialized")
    
    return storage_client.list_tasks(max_results=max_results, created_by=created_by)

def delete_task(task_id: str, storage_client=None, created_by=None):
    """
    Delete a task
    
    Args:
        task_id: ID of the task to delete
        storage_client: Storage client for persistence
        
    Returns:
        Boolean indicating success or failure
    """
    if not storage_client:
        raise ValueError("Storage client not initialized")
    
    if not get_task(task_id, storage_client=storage_client, created_by=created_by):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    success = storage_client.delete_task(task_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or delete failed")
    
    return True

# Helper function to run async tasks in background
def run_async_task(async_func, *args, **kwargs):
    """Helper function to run async functions in background tasks"""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_func(*args, **kwargs))
    finally:
        loop.close()
        
async def process_deep_research_query(
    client,
    task_id: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 10000,
    reasoning_level: str = "medium",
    storage_client = None
):
    """Process a deep research query in the background"""
    try:
        # Update the task messages
        update_task_messages(task_id, messages, storage_client=storage_client)
        
        # Perform the deep research
        response = await client.chat_completion_with_updates(
            messages=messages,
            task_id=task_id,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_search=True,  # Always enable search for deep research
            reasoning_level=reasoning_level
        )
        
        # Task status is already updated in the chat_completion_with_updates method
    except Exception as e:
        update_task_status(task_id, "failed", error=str(e), storage_client=storage_client)