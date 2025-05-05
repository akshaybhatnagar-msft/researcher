from typing import Any, Dict
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import BingGroundingTool

from dotenv import load_dotenv
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
            name="temp-search-agent",
            instructions="Perform web search and return raw results",
            tools=bing.definitions,
            headers={"x-ms-enable-preview": "true"}
        )

        # Create thread and process query
        thread = project_client.agents.create_thread()
        project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=query
        )

        # Execute search
        run = project_client.agents.create_and_process_run(
            thread_id=thread.id,
            agent_id=agent.id
        )

        if run.status == "failed":
            raise RuntimeError(f"Search failed: {run.last_error}")

        # Retrieve and process results
        messages = project_client.agents.list_messages(thread_id=thread.id)
        return _process_bing_response(messages.data[0])

def _process_bing_response(message) -> Dict[str, Any]:
    """Extract structured results from Grounding with Bing response"""
    citations = message.content[0].annotations
    return {
        "organic_results": [
            {
                "title": citation.title,
                "link": citation.url,
                "snippet": citation.text
            } for citation in citations
        ][:5]  # Limit to top 5 results
    }

def main():
    # Example usage
    query = input("Enter your search query: ")
    try:
        results = _search_with_bing_grounding(query)
        print("\nTop 5 Search Results:")
        for idx, result in enumerate(results["organic_results"], 1):
            print(f"\nResult {idx}:")
            print(f"Title: {result['title']}")
            print(f"Link: {result['link']}")
            print(f"Snippet: {result['snippet']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()