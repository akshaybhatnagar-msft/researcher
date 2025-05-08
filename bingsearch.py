from typing import Any, Dict, List
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import BingGroundingTool

from dotenv import load_dotenv
import json
import re
load_dotenv()


def _search_with_bing_grounding(query: str) -> Dict[str, Any]:
    """Perform a web search using Azure's Grounding with Bing service"""
    # Initialize Azure AI Client
    project_client = AIProjectClient.from_connection_string(
        credential=DefaultAzureCredential(),
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
        return _process_bing_response(messages.data[0])

def _process_bing_response(message) -> Dict[str, Any]:
    """Extract structured results from Grounding with Bing response"""
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
                content = " ".join([item.text for item in content if hasattr(item, 'text')])
            
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
    
def main():
    # Example usage
    query = input("Enter your search query: ")
    try:
        results = _search_with_bing_grounding(query)
        
        if not results["organic_results"]:
            print("No search results found or could not parse results.")
            return
            
        print("\nTop Search Results:")
        for idx, result in enumerate(results["organic_results"], 1):
            print(f"\nResult {idx}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Snippet: {result['snippet']}")
            
        # Demonstrate how to access the structured data programmatically
        print("\nAccess results programmatically:")
        print(f"Number of results: {len(results['organic_results'])}")
        if results["organic_results"]:
            print(f"First result title: {results['organic_results'][0]['title']}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()