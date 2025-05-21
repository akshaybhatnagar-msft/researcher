# app.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from contextlib import asynccontextmanager

import os
import uvicorn

# Import our modules
from models import QueryRequest, TaskResponse
from storage_client import TaskStorageClient
from hybrid_storage_client import HybridTaskStorageClient
from azure_client import AzureO3Client
from task_manager import (
    create_task, update_task_messages, 
    get_task, list_tasks, delete_task, run_async_task,
    process_deep_research_query
)
from auth import validate_token
from config import (
    AZURE_ENDPOINT, AZURE_API_KEY, SERP_API_KEY,
    AZURE_STORAGE_TABLE_NAME,
    USE_GROUNDING_WITH_BING, validate_config
)

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.trace import (
    get_tracer_provider,
)
from opentelemetry.propagate import extract
from logging import getLogger, INFO

if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    configure_azure_monitor()

tracer = trace.get_tracer(__name__,
                          tracer_provider=get_tracer_provider())

logger = getLogger(__name__)

# Initialize clients
storage_client = None
azure_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients on startup and clean up on shutdown"""
    global storage_client, azure_client
    
    try:
        # Validate configuration
        validate_config()
        
        # Initialize storage client with fallback option enabled
        storage_client = HybridTaskStorageClient(
            table_name=AZURE_STORAGE_TABLE_NAME
        )
        
        # Initialize Azure client
        azure_client = AzureO3Client(
            endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            serp_api_key=SERP_API_KEY,
            storage_client=storage_client
        )
        
        print("Application startup complete")
        
    except Exception as e:
        print(f"Error during startup: {e}")
        raise
    
    yield  # This is where the app runs
    
    # Cleanup code goes here (runs on shutdown)
    print("Application shutdown complete")

# Initialize the app with the lifespan context manager
app = FastAPI(
    title="Deep Research API",
    description="API for performing deep research using Azure O3 with web search capabilities and real-time updates",
    version="1.2.0",
    lifespan=lifespan
)

FastAPIInstrumentor.instrument_app(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_azure_client():
    """Get the Azure O3 client instance"""
    if not azure_client:
        raise HTTPException(
            status_code=500, 
            detail="Azure client not initialized. Please check your configuration."
        )
    
    return azure_client

def get_storage_client():
    """Get the storage client instance"""
    if not storage_client:
        raise HTTPException(
            status_code=500, 
            detail="Storage client not initialized. Please check your configuration."
        )
    
    return storage_client

# API endpoints
@app.get("/", dependencies=[Depends(validate_token)])
async def root(request: Request):
    claims = request.state.claims
    return {"message": f'Welcome to the Deep Research API. See /docs for API documentation.'}

@app.post("/api/query", dependencies=[Depends(validate_token)], response_model=TaskResponse)
async def query(
    request: Request,
    queryRequest: QueryRequest, 
    background_tasks: BackgroundTasks,
    client: AzureO3Client = Depends(get_azure_client),
    storage: HybridTaskStorageClient = Depends(get_storage_client)
):
    """Initiate a deep research query and return a task ID for polling the status"""
    try:
        created_by = request.state.claims.get("oid")
        if not created_by:
            created_by = request.state.claims.get("appid")
        # Create a new task
        task_id = create_task(storage_client=storage, created_by=created_by)
        
        # Create messages from the query and system prompt
        messages = [
            {"role": "system", "content": queryRequest.system_prompt},
            {"role": "user", "content": queryRequest.query}
        ]
        
        # Update initial task status
        update_task_messages(task_id, messages, storage_client=storage)
        
        # Launch the task in the background
        background_tasks.add_task(
            run_async_task,
            process_deep_research_query,
            client=client,
            task_id=task_id,
            messages=messages,
            temperature=queryRequest.temperature,
            max_tokens=queryRequest.max_tokens,
            reasoning_level=queryRequest.reasoning_level,
            storage_client=storage
        )
        
        # Return the task ID and status URL for polling immediately
        return {
            "task_id": task_id,
            "status_url": f"/api/status/{task_id}"
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks", dependencies=[Depends(validate_token)])
async def get_tasks(request: Request, storage: HybridTaskStorageClient = Depends(get_storage_client)):
    """Get the list of all tasks"""
    created_by = request.state.claims.get("oid")
    if not created_by:
        created_by = request.state.claims.get("appid")

    logger.info(f"Fetching tasks for user: {created_by}")
    try:
        # Fetch tasks from storage
        tasks = list_tasks(storage_client=storage, created_by=created_by)
        return {
            "tasks": [
                {
                    "task_id": task.get("RowKey"),
                    "status": task.get("status"),
                    "progress": task.get("progress"),
                    "created_at": task.get("created_at"),
                    "updated_at": task.get("updated_at")
                }
                for task in tasks
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))  

@app.get("/api/thought-process/{task_id}", dependencies=[Depends(validate_token)])
async def get_task_thought_process(
    request: Request,
    task_id: str, 
    storage: HybridTaskStorageClient = Depends(get_storage_client)
):
    created_by = request.state.claims.get("oid")
    if not created_by:
        created_by = request.state.claims.get("appid")
    """Get the thought process of a task"""
    task = get_task(task_id, storage_client=storage, created_by=created_by)
    
    return {
        "task_id": task.get("RowKey"),
        "status": task.get("status"),
        "progress": task.get("progress"),
        "thought_process": task.get("thought_process", []),
        "result": task.get("result"),
        "tool_calls": task.get("tool_calls", []),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at")
    }

@app.get("/api/status/{task_id}", dependencies=[Depends(validate_token)])
async def get_task_status(
    request: Request,
    task_id: str, 
    storage: HybridTaskStorageClient = Depends(get_storage_client)
):
    """Get the status of a task"""
    task = get_task(task_id, storage_client=storage)
    
    return {
        "task_id": task.get("RowKey"),
        "status": task.get("status"),
        "progress": task.get("progress"),
        "thought_process": task.get("thought_process", []),
        "result": task.get("result"),
        "tool_calls": task.get("tool_calls", []),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at")
    }

@app.delete("/api/tasks/{task_id}", dependencies=[Depends(validate_token)])
async def delete_task_endpoint(
    request: Request,
    task_id: str,
    storage: HybridTaskStorageClient = Depends(get_storage_client)
):
    """Delete a task"""
    delete_task(task_id, storage_client=storage)
    return {"message": f"Task {task_id} deleted"}

# Health check endpoint (useful for Azure)
@app.get("/health")
async def health_check():
    global storage_client, azure_client
    
    status = {
        "status": "healthy",
        "USE_GROUNDING_WITH_BING": USE_GROUNDING_WITH_BING,
        "storage_client_initialized": storage_client is not None,
        "azure_client_initialized": azure_client is not None
    }
    
    return status

# Run the server if executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)