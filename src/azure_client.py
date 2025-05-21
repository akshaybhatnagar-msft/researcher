# azure_client.py
import json
import requests
import uuid
from datetime import datetime
from typing import Dict, List, Any
import os
from fastapi import HTTPException

class AzureO3Client:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2025-01-01-preview",
        serp_api_key: str = None,
        storage_client = None
    ):
        """
        Initialize an Azure O3 client with SERP API search capability
        
        Args:
            endpoint: Azure AI Foundry endpoint URL
            api_key: Azure AI Foundry API key
            api_version: API version to use
            serp_api_key: API key for SERP API (optional)
            storage_client: HybridTaskStorageClient instance for persistence
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.api_version = api_version
        self.serp_api_key = serp_api_key
        self.storage_client = storage_client
        self.use_grounding_with_bing = os.environ.get("USE_GROUNDING_WITH_BING", "False") == "True"
        
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers required for API calls"""
        return {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }

    
    def _search_with_bing_grounding(self, query: str) -> Dict[str, Any]:
        """Perform a web search using Azure's Grounding with Bing service"""
        from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
        from azure.ai.projects import AIProjectClient
        from azure.ai.projects.models import BingGroundingTool
        import re
        
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

            # Initialize Azure AI Client
            project_client = AIProjectClient.from_connection_string(
                credential= credential,
                conn_str="eastus2.api.azureml.ms;7999d76d-a4cb-40e9-b3db-73a306deca7f;akbhatna-deep;deepa1"
            )
            
            # Get Bing connection and initialize tool
            bing_connection = project_client.connections.get(
                connection_name="bingGrounding"
            )
            bing = BingGroundingTool(connection_id=bing_connection.id)

            # Create agent with Bing grounding
            with project_client:
                agent = project_client.agents.create_agent(
                    model="gpt-4o",
                    name="structured-search-agent",
                    instructions="""
                    Perform a web search and return results in a structured JSON format as shown below:
                    
                    ```json
                    {
                        "results": [
                            {
                                "title": "Result Title",
                                "url": "https://result-url.com",
                                "snippet": "Short description of the result"
                            }
                        ]
                    }
                    ```
                    
                    Include the title, URL, and snippet for each result.
                    Limit to the top 5 most relevant results.
                    IMPORTANT: Always format your response as valid JSON inside triple backticks.
                    """,
                    tools=[*bing.definitions],
                    headers={"x-ms-enable-preview": "true"}
                )

                # Create thread and process query
                thread = project_client.agents.create_thread()
                project_client.agents.create_message(
                    thread_id=thread.id,
                    role="user",
                    content=f"""Search for: {query}
                    
                    IMPORTANT: Format your response as valid JSON inside triple backticks with the following structure:
                    {{
                        "results": [
                            {{
                                "title": "Result Title",
                                "url": "https://result-url.com",
                                "snippet": "Short description of the result"
                            }}
                        ]
                    }}
                    """
                )

                # Execute search
                run = project_client.agents.create_and_process_run(
                    thread_id=thread.id,
                    agent_id=agent.id,
                    additional_instructions="Always format your response as valid JSON inside triple backticks."
                )

                if run.status == "failed":
                    raise RuntimeError(f"Search failed: {run.last_error}")

                # Retrieve and process results
                messages = project_client.agents.list_messages(thread_id=thread.id)
                return self._process_bing_response(messages.data[0])

        except Exception as e:
            print(f"Error in Bing search: {e}")
            raise HTTPException(status_code=500, detail=f"Bing search failed: {str(e)}")
    
    def _process_bing_response(self, message) -> Dict[str, Any]:
        """Extract structured results from Grounding with Bing response"""
        import re
        import json
        
        try:
            # First check if the message has list-type content (newer API versions)
            if hasattr(message, 'content') and isinstance(message.content, list):
                content_list = message.content
                
                for item in content_list:
                    if hasattr(item, 'text'):
                        # Extract text content
                        text = item.text.value
                        # Look for JSON in code blocks
                        json_matches = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
                        
                        for json_match in json_matches:
                            try:
                                parsed_json = json.loads(json_match)
                                if 'results' in parsed_json:
                                    # Transform to our output format
                                    organic_results = []
                                    for result in parsed_json.get('results', [])[:5]:
                                        organic_results.append({
                                            "title": result.get("title", ""),
                                            "url": result.get("url", ""),
                                            "snippet": result.get("snippet", "")
                                        })
                                    return {"organic_results": organic_results}
                            except json.JSONDecodeError:
                                pass
                        
                        # Try to find JSON-like structure in plain text
                        try:
                            # Find JSON object in text
                            json_start = text.find('{')
                            json_end = text.rfind('}')
                            if json_start != -1 and json_end != -1 and json_end > json_start:
                                json_text = text[json_start:json_end+1]
                                parsed_json = json.loads(json_text)
                                if 'results' in parsed_json:
                                    # Transform to our output format
                                    organic_results = []
                                    for result in parsed_json.get('results', [])[:5]:
                                        organic_results.append({
                                            "title": result.get("title", ""),
                                            "url": result.get("url", ""),
                                            "snippet": result.get("snippet", "")
                                        })
                                    return {"organic_results": organic_results}
                            else:
                                # Try to parse the entire text as JSON
                                parsed_json = json.loads(text)
                                if 'results' in parsed_json:
                                    # Transform to our output format
                                    organic_results = []
                                    for result in parsed_json.get('results', [])[:5]:
                                        organic_results.append({
                                            "title": result.get("title", ""),
                                            "url": result.get("url", ""),
                                            "snippet": result.get("snippet", "")
                                        })
                                    return {"organic_results": organic_results}
                        except json.JSONDecodeError:
                            pass
                            
            # Alternative handling for string content (older API versions)
            elif hasattr(message, 'content') and isinstance(message.content, str):
                content = message.content
                
                # Try parsing the entire content as JSON first
                try:
                    parsed_json = json.loads(content)
                    if 'results' in parsed_json:
                        organic_results = []
                        for result in parsed_json.get('results', [])[:5]:
                            organic_results.append({
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "snippet": result.get("snippet", "")
                            })
                        return {"organic_results": organic_results}
                except json.JSONDecodeError:
                    pass
                
                # Look for JSON in code blocks
                json_matches = re.findall(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                for json_match in json_matches:
                    try:
                        parsed_json = json.loads(json_match)
                        if 'results' in parsed_json:
                            organic_results = []
                            for result in parsed_json.get('results', [])[:5]:
                                organic_results.append({
                                    "title": result.get("title", ""),
                                    "url": result.get("url", ""),
                                    "snippet": result.get("snippet", "")
                                })
                            return {"organic_results": organic_results}
                    except json.JSONDecodeError:
                        pass
                
                # Try to find JSON-like structure in plain text
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_text = content[json_start:json_end+1]
                        parsed_json = json.loads(json_text)
                        if 'results' in parsed_json:
                            organic_results = []
                            for result in parsed_json.get('results', [])[:5]:
                                organic_results.append({
                                    "title": result.get("title", ""),
                                    "url": result.get("url", ""),
                                    "snippet": result.get("snippet", "")
                                })
                            return {"organic_results": organic_results}
                except json.JSONDecodeError:
                    pass
                    
            # If no JSON structure found, try to extract search results using regex
            if hasattr(message, 'content'):
                content = message.content
                if isinstance(content, list):
                    # Join all text items
                    content = " ".join([item.text.value for item in content if hasattr(item, 'text')])
                
                # Regular expressions to extract title, URL and snippet patterns
                result_pattern = r'(?:Title|title):\s*([^\n]+).*?(?:URL|url|Url|Link|link):\s*([^\n]+).*?(?:Snippet|snippet|Description|description):\s*([^\n]+)'
                matches = re.findall(result_pattern, content, re.DOTALL)
                
                if matches:
                    organic_results = []
                    for match in matches[:5]:  # Limit to top 5
                        organic_results.append({
                            "title": match[0].strip(),
                            "url": match[1].strip(),
                            "snippet": match[2].strip()
                        })
                    return {"organic_results": organic_results}
            
            # If everything fails, return empty result
            print("Warning: Could not extract structured data from response")
            return {"organic_results": []}
            
        except Exception as e:
            print(f"Error processing response: {e}")
            raise RuntimeError(f"Failed to process Bing response: {e}")
    
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
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_details = {
                "status_code": response.status_code,
                "response_text": response.text,
                "url": response.url
            }
            raise HTTPException(
                status_code=response.status_code,
                detail=f"HTTP error occurred: {e}. Details: {error_details}"
            )
        
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
                if self.use_grounding_with_bing:
                    search_results = self._search_with_bing_grounding(args["query"])                    
                else:
                    search_results = self._search_with_serp(args["query"])                
                
                # Extract relevant information from search results
                organic_results = search_results.get("organic_results", [])
                simplified_results = [
                    {
                        "title": result.get("title"),
                        "link": result.get("link", result.get("url")),
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
        if self.storage_client:
            # Store thought in Table Storage
            result = self.storage_client.add_thought_to_task(task_id, thought)
            if not result:
                raise HTTPException(status_code=500, detail=f"Failed to add thought to task {task_id}")
        else:
            raise HTTPException(status_code=500, detail="Storage client not initialized")

    def add_tool_call_to_task(self, task_id: str, tool_call_data: Dict[str, Any]) -> None:
        """Add a tool call to the task"""
        if self.storage_client:
            # Store tool call in Table Storage
            result = self.storage_client.add_tool_call_to_task(task_id, tool_call_data)
            if not result:
                raise HTTPException(status_code=500, detail=f"Failed to add tool call to task {task_id}")
        else:
            raise HTTPException(status_code=500, detail="Storage client not initialized")
    
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
        from task_manager import update_task_status, update_task_messages
        
        url = f"{self.endpoint}/openai/deployments/o3/chat/completions?api-version={self.api_version}"
        
        # Update task status to in_progress if not already
        task = self.storage_client.get_task(task_id)
        if task and task.get("status") == "pending":
            update_task_status(task_id, "in_progress", progress=0.1, storage_client=self.storage_client)
        
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
                payload["reasoning_effort"] = "high"
                # # Add explicit instructions for detailed reasoning
                # if has_system_message:
                #     for i, msg in enumerate(messages):
                #         if msg["role"] == "system":
                #             messages[i]["content"] += "\nPlease use extensive step-by-step reasoning, consider multiple perspectives, and be thorough in your analysis."
                #             break
                # else:
                #     messages.insert(0, {
                #         "role": "system", 
                #         "content": "Please use extensive step-by-step reasoning, consider multiple perspectives, and be thorough in your analysis."
                #     })
            elif reasoning_level == "medium":
                payload["reasoning_effort"] = "medium"
            else:
                payload["reasoning_effort"] = "low"
        
        if enable_search:
            payload["tools"] = [self._create_search_tool()]
            payload["tool_choice"] = "auto"
        
        # Update task with the current messages
        update_task_messages(task_id, messages, storage_client=self.storage_client)
        update_task_status(task_id, "in_progress", storage_client=self.storage_client)
        
        # Make the streaming API request
        try:
            response = requests.post(
                url, 
                headers=self._get_headers(), 
                json=payload,
                stream=True  # Enable streaming at the request level
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_details = {
                "status_code": response.status_code,
                "response_text": response.text,
                "url": response.url
            }
            raise HTTPException(
                status_code=response.status_code,
                detail=f"HTTP error occurred: {e}. Details: {error_details}"
            )
        
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
            update_task_status(task_id, "in_progress", progress=0.7, storage_client=self.storage_client)
            self.add_thought_to_task(task_id, "Processing tool calls to retrieve additional information...")
            
            tool_results = self._handle_tool_calls(tool_calls)
            
            # Add the tool results to the messages and make another API call
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": final_response["choices"][0]["message"]["tool_calls"]                
            })
            update_task_messages(task_id, messages, storage_client=self.storage_client)
            
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
            update_task_status(task_id, "in_progress", progress=0.7, storage_client=self.storage_client)
            update_task_messages(task_id, messages, storage_client=self.storage_client)
            
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
            update_task_status(task_id, "in_progress", progress=0.9, storage_client=self.storage_client)
            self.add_thought_to_task(task_id, "Finalizing response and completing research task...")
            
        # Update task with final results
        update_task_status(task_id, "completed", progress=1.0, result=final_response, storage_client=self.storage_client)
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