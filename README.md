# Deep Research API

A FastAPI-based service for performing deep research using Azure OpenAI with web search capabilities and real-time updates.

## Architecture

1. **Persistent Storage**: Uses Azure Table Storage for task persistence
2. **Modular Design**: Code is organized into separate modules for better maintainability
3. **Authentication**: Uses Azure AD authentication with JSON Web Tokens (JWT)
4. **Background Processing**: Long-running tasks are processed asynchronously
5. **Modern FastAPI Patterns**: Uses the FastAPI lifespan context manager for application lifecycle management

## Project Structure

- `app.py` - Main FastAPI application
- `models.py` - Pydantic data models
- `storage_client.py` - Azure Table Storage client for task persistence
- `azure_client.py` - Azure OpenAI client for AI requests
- `task_manager.py` - Task management utilities
- `auth.py` - Authentication utilities
- `config.py` - Application configuration

## Requirements

- Python 3.8+
- Azure Table Storage account
- Azure OpenAI service
- SERP API key (or use Azure's Grounding with Bing)

## Environment Variables

Create a `.env` file with the following variables:

```
# Azure OpenAI configuration
AZURE_ENDPOINT=your-azure-openai-endpoint
AZURE_API_KEY=your-azure-openai-key

# Azure Table Storage configuration
AZURE_STORAGE_TABLE_URL=your-table-storage-account-url
AZURE_STORAGE_TABLE_NAME=DeepResearchTasks

# Azure AD configuration (for authentication)
AZURE_TENANT_ID=your-azure-tenant-id
AZURE_AD_APP_ID=your-app-id

# Search configuration
SERP_API_KEY=your-serp-api-key
USE_GROUNDING_WITH_BING=False
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables
4. Run the application: `python app.py`

## API Endpoints

- `GET /` - Root endpoint
- `POST /api/query` - Start a new research query
- `GET /api/tasks` - List all tasks
- `GET /api/status/{task_id}` - Get task status
- `GET /api/thought-process/{task_id}` - Get detailed thought process for a task
- `DELETE /api/tasks/{task_id}` - Delete a task
- `GET /health` - Health check endpoint

## Authentication

The API uses Azure AD authentication. Include a valid Azure AD JWT token in the `Authorization` header of your requests:

```
Authorization: Bearer <your-token>
```

## Scaling Considerations

- The application now uses Azure Table Storage for task persistence, allowing it to scale horizontally
- Tasks are processed asynchronously in the background
- Azure Table Storage provides durable, low-cost storage for task data