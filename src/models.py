# models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

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

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "in_progress", "completed", "failed"
    progress: float = 0.0  # 0.0 to 1.0
    messages: List[Dict[str, Any]] = []
    thought_process: List[Dict[str, Any]] = []  # Field to store reasoning steps
    tool_calls: List[Dict[str, Any]] = []       # Field to track tool calls
    result: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str
    error: Optional[str] = None

    @classmethod
    def from_storage_entity(cls, entity):
        """Create a TaskStatus instance from a storage entity"""
        return cls(
            task_id=entity.get("RowKey"),
            status=entity.get("status", "pending"),
            progress=entity.get("progress", 0.0),
            messages=entity.get("messages", []),
            thought_process=entity.get("thought_process", []),
            tool_calls=entity.get("tool_calls", []),
            result=entity.get("result"),
            created_at=entity.get("created_at", datetime.now().isoformat()),
            updated_at=entity.get("updated_at", datetime.now().isoformat()),
            error=entity.get("error")
        )