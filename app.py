# app.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import json
import requests
import uvicorn
import uuid
from datetime import datetime
from dotenv import load_dotenv


from authlib.jose import jwt, JoseError
from authlib.jose import JsonWebKey

import requests

# Load environment variables
load_dotenv()

# Azure AD configuration
AZURE_AD_TENANT_ID = os.environ.get("AZURE_TENANT_ID")
AZURE_AD_APP_ID = os.environ.get("AZURE_AD_APP_ID")  # API app ID (audience)
AZURE_AD_AUDIENCE = f"api://{AZURE_AD_APP_ID}"
AZURE_AD_ISSUER = f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/v2.0"

AZURE_ISSUER = f"https://sts.windows.net/{AZURE_AD_TENANT_ID}/"
AZURE_JWKS_URI = f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/discovery/v2.0/keys"
AZURE_AUDIENCE = f"api://{AZURE_AD_APP_ID}"

# Fetch JWKS once (cache this in production)
JWKS = JsonWebKey.import_key_set(requests.get(AZURE_JWKS_URI).json())

# Bearer token parser
bearer_scheme = HTTPBearer()

# Token validation
async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials

    try:
        claims = jwt.decode(
            token,
            key=JWKS,
            claims_options={
                "iss": {"values": [AZURE_ISSUER]},
                # "aud": {"values": [AZURE_AUDIENCE]},
            }
        )
        claims.validate()
        return claims  # contains `appid`, `azp`, etc.

    except JoseError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "in_progress", "completed", "failed"
    progress: float = 0.0  # 0.0 to 1.0
    messages: List[Dict[str, Any]] = []
    thought_process: List[Dict[str, Any]] = []  # New field to store reasoning steps
    tool_calls: List[Dict[str, Any]] = []       # New field to track tool calls
    result: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str
    error: Optional[str] = None

# In-memory storage for tasks (replace with Redis or database in production)
tasks = {}

class AzureO3Client:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2025-01-01-preview",
        serp_api_key: str = None
    ):
        """
        Initialize an Azure O3 client with SERP API search capability
        
        Args:
            endpoint: Azure AI Foundry endpoint URL
            api_key: Azure AI Foundry API key
            api_version: API version to use
            serp_api_key: API key for SERP API (optional)
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.api_version = api_version
        self.serp_api_key = serp_api_key
        
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers required for API calls"""
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
    
    def _search_with_serp(self, query: str) -> Dict[str, Any]:
        """Perform a web search using SERP API"""
        if not self.serp_api_key:
            raise ValueError("SERP API key not provided")
            
        url = "https://serpapi.com/search"
        params = {
            "api_key": self.serp_api_key,
            "q": query,
            "engine": "google"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def _create_search_tool(self) -> Dict[str, Any]:
        """Create a search tool definition for the model"""
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and execute tool calls from the model"""
        tool_results = []
        
        for tool_call in tool_calls:
            if tool_call["function"]["name"] == "search":
                args = json.loads(tool_call["function"]["arguments"])
                search_results = self._search_with_serp(args["query"])
                
                # Extract relevant information from search results
                organic_results = search_results.get("organic_results", [])
                simplified_results = [
                    {
                        "title": result.get("title"),
                        "link": result.get("link"),
                        "snippet": result.get("snippet")
                    }
                    for result in organic_results[:5]  # Limit to top 5 results
                ]
                
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "output": json.dumps(simplified_results)
                })
            
        return tool_results
    
    def add_thought_to_task(self, task_id: str, thought: str) -> None:
        """Add a thought process step to the task"""
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        timestamp = datetime.now().isoformat()
        
        # Add the thought to the task's thought_process list
        tasks[task_id].thought_process.append({
            "timestamp": timestamp,
            "content": thought
        })
        
        # Update the task's timestamp
        tasks[task_id].updated_at = timestamp

    def add_tool_call_to_task(self, task_id: str, tool_call_data: Dict[str, Any]) -> None:
        """Add a tool call to the task"""
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        timestamp = datetime.now().isoformat()
        
        # Add the tool call to the task's tool_calls list
        # Build a map of existing tool calls by ID
        existing_calls = {
            call["data"][0]["id"]: call
            for call in tasks[task_id].tool_calls
            if call["data"]
        }

        # Now merge/overwrite with the new ones
        for call in tool_call_data:
            call_id = call["id"]
            existing_calls[call_id] = {
                "timestamp": timestamp,
                "data": [call]
            }

        # Replace the tool_calls list with deduplicated values
        tasks[task_id].tool_calls = list(existing_calls.values())
        
        # Update the task's timestamp
        tasks[task_id].updated_at = timestamp
    
    async def chat_completion_with_updates(
        self,
        messages: List[Dict[str, str]],
        task_id: str,
        temperature: float = 1,
        max_tokens: int = 10000,
        enable_search: bool = True,
        reasoning_level: str = "none"
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the O3 model with streaming for thought process
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            task_id: ID of the task to update
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            enable_search: Whether to enable the search tool
            reasoning_level: Level of reasoning to use ('none', 'low', 'medium', 'high')
            
        Returns:
            The model's response
        """
        url = f"{self.endpoint}/openai/deployments/o3/chat/completions?api-version={self.api_version}"
        
        # Update task status to in_progress if not already
        if tasks[task_id].status == "pending":
            update_task_status(task_id, "in_progress", progress=0.1)
        
        # Add a system instruction to include reasoning steps
        has_system_message = any(msg["role"] == "system" for msg in messages)
        reasoning_instruction = (
            "Please think step by step and explain your reasoning process as you work. "
            "When you use tools, explain why you are using them and what you hope to learn. "
            "Share your thought process and methodology throughout."
        )
        
        if has_system_message:
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    messages[i]["content"] += f"\n{reasoning_instruction}"
                    break
        else:
            messages.insert(0, {"role": "system", "content": reasoning_instruction})
        
        # Set up the payload with streaming enabled
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True  # Enable streaming
        }
        
        # Add reasoning_level if supported by the API
        if reasoning_level and reasoning_level != "none":
            if reasoning_level == "high":
                # Add explicit instructions for detailed reasoning
                if has_system_message:
                    for i, msg in enumerate(messages):
                        if msg["role"] == "system":
                            messages[i]["content"] += "\nPlease use extensive step-by-step reasoning, consider multiple perspectives, and be thorough in your analysis."
                            break
                else:
                    messages.insert(0, {
                        "role": "system", 
                        "content": "Please use extensive step-by-step reasoning, consider multiple perspectives, and be thorough in your analysis."
                    })
        
        if enable_search:
            payload["tools"] = [self._create_search_tool()]
            payload["tool_choice"] = "auto"
        
        # Update task with the current messages
        update_task_messages(task_id, messages)
        update_task_status(task_id, "in_progress", progress=0.3)
        
        # Make the streaming API request
        response = requests.post(
            url, 
            headers=self._get_headers(), 
            json=payload,
            stream=True  # Enable streaming at the request level
        )
        response.raise_for_status()
        
        # Initialize variables to collect the response
        current_thought = ""
        full_response = ""
        collected_chunks = []
        finish_reason = None
        
        # Dictionary to track ongoing tool calls by index
        current_tool_calls = {}
        
        # Process the streaming response
        for line in response.iter_lines():
            if line:
                # Skip the "data: " prefix
                if line.startswith(b"data: "):
                    line = line[6:]
                    
                if line.strip() == b"[DONE]":
                    break
                    
                try:
                    chunk = json.loads(line)
                    collected_chunks.append(chunk)
                    
                    # Extract delta content
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        
                        # Extract content from delta
                        if "content" in delta and delta["content"]:
                            content = delta["content"]
                            current_thought += content
                            full_response += content
                            
                            # Update thought process every few tokens or when hitting punctuation
                            if content.endswith((".", "!", "?", "\n")) or len(current_thought) > 100:
                                self.add_thought_to_task(task_id, current_thought)
                                current_thought = ""
                        
                        # Check for tool calls in the delta
                        if "tool_calls" in delta:
                            # Process and aggregate tool calls
                            for tool_call_delta in delta["tool_calls"]:
                                tool_call_index = tool_call_delta.get("index", 0)
                                
                                # Initialize this tool call if first time seeing it
                                if tool_call_index not in current_tool_calls:
                                    current_tool_calls[tool_call_index] = {
                                        "id": tool_call_delta.get("id", str(uuid.uuid4())),
                                        "type": "function",
                                        "function": {
                                            "name": "",
                                            "arguments": ""
                                        }
                                    }
                                
                                # Update with new information
                                if "function" in tool_call_delta:
                                    function_delta = tool_call_delta["function"]
                                    
                                    if "name" in function_delta:
                                        current_tool_calls[tool_call_index]["function"]["name"] = function_delta["name"]
                                    
                                    if "arguments" in function_delta:
                                        current_tool_calls[tool_call_index]["function"]["arguments"] += function_delta["arguments"]
                                
                                # Add the tool call to the task when complete or significant updates
                                if "id" in tool_call_delta or len(current_tool_calls) > 2:
                                    # Convert to list for storage
                                    tool_calls_list = list(current_tool_calls.values())
                                    self.add_tool_call_to_task(task_id, tool_calls_list)
                        
                        # Check if this is the end of the response
                        if "finish_reason" in chunk["choices"][0]:
                            finish_reason = chunk["choices"][0]["finish_reason"]
                            
                            # Add final tool calls to task if any
                            if current_tool_calls:
                                tool_calls_list = list(current_tool_calls.values())
                                self.add_tool_call_to_task(task_id, tool_calls_list)
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
        
        # Add any remaining thought content
        if current_thought:
            self.add_thought_to_task(task_id, current_thought)
        
        # Construct the final response from the collected chunks
        final_response = self._construct_final_response(collected_chunks)
        
        # Handle tool calls if present and finish_reason is "tool_calls"
        if finish_reason == "tool_calls" and "tool_calls" in final_response["choices"][0]["message"]:
            tool_calls = final_response["choices"][0]["message"]["tool_calls"]
            
            # Update status to show we're processing tool calls
            update_task_status(task_id, "in_progress", progress=0.6)
            self.add_thought_to_task(task_id, "Processing tool calls to retrieve additional information...")
            
            tool_results = self._handle_tool_calls(tool_calls)
            
            # Add the tool results to the messages and make another API call
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": final_response["choices"][0]["message"]["tool_calls"]                
            })
            update_task_messages(task_id, messages)
            
            for tool_result in tool_results:
                tool_result_content = tool_result["output"]
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": tool_result_content
                })
                
                # Add information about the tool results to the thought process
                self.add_thought_to_task(
                    task_id, 
                    f"Received results from tool call. Processing information from search results..."
                )
            
            # Update status to show we're continuing with tool results
            update_task_status(task_id, "in_progress", progress=0.7)
            update_task_messages(task_id, messages)
            
            # Make another call with the tool responses
            return await self.chat_completion_with_updates(
                messages=messages,
                task_id=task_id,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_search=enable_search,
                reasoning_level=reasoning_level
            )
        else:
            # Update progress before completing
            update_task_status(task_id, "in_progress", progress=0.9)
            self.add_thought_to_task(task_id, "Finalizing response and completing research task...")
            
        # Update task with final results
        update_task_status(task_id, "completed", progress=1.0, result=final_response)
        return final_response
    
    def _construct_final_response(self, chunks):
        """Construct a final response object from streaming chunks"""
        if not chunks:
            return {"choices": [{"message": {"content": ""}}]}
        
        # The last non-empty chunk should contain the complete message
        final_chunk = None
        for chunk in reversed(chunks):
            if chunk.get("choices") and len(chunk["choices"]) > 0:
                final_chunk = chunk
                break
        
        if not final_chunk:
            return {"choices": [{"message": {"content": ""}}]}
        
        # Reconstruct the full content from all chunks
        full_content = ""
        tool_calls = []
        tool_call_map = {}  # Map to track tool calls by index
        
        # First pass: gather all content and prepare tool calls structure
        for chunk in chunks:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                
                # Add content
                if "content" in delta and delta["content"]:
                    full_content += delta["content"]
                
                # Process tool calls
                if "tool_calls" in delta:
                    for tool_call_delta in delta["tool_calls"]:
                        # Get the index for this tool call
                        tool_call_id = tool_call_delta.get("index", 0)
                        
                        # Create a new tool call if we haven't seen this index before
                        if tool_call_id not in tool_call_map:
                            new_id = str(uuid.uuid4())
                            tool_call_map[tool_call_id] = {
                                "id": new_id,
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                },
                                "type": "function"  # Add required type field
                            }
                        
                        # Update function information if present
                        if "function" in tool_call_delta:
                            function_delta = tool_call_delta["function"]
                            
                            if "name" in function_delta:
                                tool_call_map[tool_call_id]["function"]["name"] = function_delta["name"]
                            
                            if "arguments" in function_delta:
                                tool_call_map[tool_call_id]["function"]["arguments"] += function_delta["arguments"]
                        
                        # Update ID if provided
                        if "id" in tool_call_delta:
                            tool_call_map[tool_call_id]["id"] = tool_call_delta["id"]
        
        # Convert the tool call map to a list
        tool_calls = list(tool_call_map.values())
        
        # Ensure argument strings are valid JSON
        for tool_call in tool_calls:
            try:
                # Attempt to parse and re-stringify to ensure valid JSON
                args_str = tool_call["function"]["arguments"]
                if args_str:
                    parsed_args = json.loads(args_str)
                    tool_call["function"]["arguments"] = json.dumps(parsed_args)
            except json.JSONDecodeError:
                # If not valid JSON, keep the string as is, the API will handle errors
                pass
        
        # Construct the final message
        final_message = {"content": full_content}
        if tool_calls:
            final_message["tool_calls"] = tool_calls
        
        # Replace the message in the final chunk
        final_chunk["choices"][0]["message"] = final_message
        
        return final_chunk

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1,
        max_tokens: int = 10000,
        enable_search: bool = True,
        reasoning_level: str = "none"
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the O3 model with optional tool calling
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            enable_search: Whether to enable the search tool
            reasoning_level: Level of reasoning to use ('none', 'low', 'medium', 'high')
            
        Returns:
            The model's response
        """
        url = f"{self.endpoint}/openai/deployments/o3/chat/completions?api-version={self.api_version}"
        
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens            
        }
        
        if enable_search:
            payload["tools"] = [self._create_search_tool()]
            payload["tool_choice"] = "auto"
        
        response = requests.post(url, headers=self._get_headers(), json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Handle any tool calls in the response
        if "tool_calls" in result["choices"][0]["message"]:
            tool_calls = result["choices"][0]["message"]["tool_calls"]
            tool_results = self._handle_tool_calls(tool_calls)
            
            # Add the tool results to the messages and make another API call
            messages.append(result["choices"][0]["message"])
            
            for tool_result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": tool_result["output"]
                })
            
            # Make another call with the tool responses
            return self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_search=enable_search,
                reasoning_level=reasoning_level
            )
            
        return result

# Get the client instance based on environment variables
def get_azure_client():
    azure_endpoint = os.environ.get("AZURE_ENDPOINT")
    azure_api_key = os.environ.get("AZURE_API_KEY")
    serp_api_key = os.environ.get("SERP_API_KEY")
    
    if not all([azure_endpoint, azure_api_key, serp_api_key]):
        raise HTTPException(
            status_code=500, 
            detail="Missing required environment variables. Please set AZURE_ENDPOINT, AZURE_API_KEY, and SERP_API_KEY."
        )
    
    return AzureO3Client(
        endpoint=azure_endpoint,
        api_key=azure_api_key,
        serp_api_key=serp_api_key
    )

# Helper functions for task management
def create_task() -> str:
    """Create a new task and return its ID"""
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        progress=0.0,
        messages=[],
        thought_process=[],
        tool_calls=[],
        created_at=timestamp,
        updated_at=timestamp
    )
    return task_id

def update_task_status(task_id: str, status: str, progress: float = None, result: Dict[str, Any] = None, error: str = None) -> None:
    """Update a task's status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    timestamp = datetime.now().isoformat()
    tasks[task_id].updated_at = timestamp
    tasks[task_id].status = status
    
    if progress is not None:
        tasks[task_id].progress = progress
    
    if result is not None:
        tasks[task_id].result = result
    
    if error is not None:
        tasks[task_id].error = error

def update_task_messages(task_id: str, messages: List[Dict[str, Any]]) -> None:
    """Update a task's messages"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    timestamp = datetime.now().isoformat()
    tasks[task_id].updated_at = timestamp
    tasks[task_id].messages = messages

# Request models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=10000, gt=0)
    enable_search: bool = Field(default=True)
    reasoning_level: str = Field(default="none", pattern="^(none|low|medium|high)$")

class QueryRequest(BaseModel):
    query: str
    system_prompt: Optional[str] = "You are a helpful assistant with access to web search capabilities."
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=10000, gt=0)
    reasoning_level: str = Field(default="medium", pattern="^(none|low|medium|high)$")

class TaskResponse(BaseModel):
    task_id: str
    status_url: str

app = FastAPI(
    title="Deep Research API",
    description="API for performing deep research using Azure O3 with web search capabilities and real-time updates",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints
@app.get("/", dependencies=[Depends(validate_token)])
async def root():
    return {"message": "Welcome to the Deep Research API. See /docs for API documentation."}

@app.post("/api/query", dependencies=[Depends(validate_token)], response_model=TaskResponse)
async def query(
    request: QueryRequest, 
    background_tasks: BackgroundTasks,
    client: AzureO3Client = Depends(get_azure_client)
):
    """Initiate a deep research query and return a task ID for polling the status"""
    try:
        # Create a new task
        task_id = create_task()
        
        # Create messages from the query and system prompt
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.query}
        ]
        
        # Update initial task status
        update_task_messages(task_id, messages)
        
        # Launch the task in the background
        background_tasks.add_task(
            run_async_task,
            process_deep_research_query,
            client=client,
            task_id=task_id,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            reasoning_level=request.reasoning_level
        )
        
        # Return the task ID and status URL for polling immediately
        return {
            "task_id": task_id,
            "status_url": f"/api/status/{task_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/api/thought-process/{task_id}", dependencies=[Depends(validate_token)])
async def get_task_thought_process(task_id: str):
    """Get the thought process of a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return {
        "task_id": task_id,
        "status": tasks[task_id].status,
        "progress": tasks[task_id].progress,
        "thought_process": tasks[task_id].thought_process,
        "tool_calls": tasks[task_id].tool_calls,
        "created_at": tasks[task_id].created_at,
        "updated_at": tasks[task_id].updated_at
    }

@app.get("/api/status/{task_id}", dependencies=[Depends(validate_token)])
async def get_task_status(task_id: str):
    """Get the status of a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return tasks[task_id]

@app.delete("/api/tasks/{task_id}", dependencies=[Depends(validate_token)])
async def delete_task(task_id: str):
    """Delete a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    del tasks[task_id]
    return {"message": f"Task {task_id} deleted"}

# Background task processor
async def process_deep_research_query(
    client: AzureO3Client,
    task_id: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    max_tokens: int = 10000,
    reasoning_level: str = "medium"
):
    """Process a deep research query in the background"""
    try:
        # Update the task messages
        update_task_messages(task_id, messages)
        
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
        update_task_status(task_id, "failed", error=str(e))

# Health check endpoint (useful for Azure)
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the server if executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)