# config.py
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

# Load environment variables
load_dotenv()

# Azure AI configuration
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
SERP_API_KEY = os.environ.get("SERP_API_KEY")

# Azure Storage configuration
AZURE_STORAGE_TABLE_URL = os.environ.get("AZURE_STORAGE_TABLE_URL")
AZURE_STORAGE_TABLE_NAME = os.environ.get("AZURE_STORAGE_TABLE_NAME", "DeepResearchTasks")

# Feature flags
USE_GROUNDING_WITH_BING = os.environ.get("USE_GROUNDING_WITH_BING", "False") == "True"

# Default credential for Azure
def get_azure_credential():
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

def validate_config():
    """Validate that required configuration is present"""
    missing = []
    warnings = []
    
    # Check for required Azure AI config
    if not AZURE_ENDPOINT:
        missing.append("AZURE_ENDPOINT")
    if not AZURE_API_KEY:
        missing.append("AZURE_API_KEY")
    if not SERP_API_KEY:
        missing.append("SERP_API_KEY")
    
    # Check for required Azure Storage config
    if not AZURE_STORAGE_TABLE_URL:
        missing.append("AZURE_STORAGE_TABLE_URL")
    else:
        # Verify that the account URL has the expected format
        if not (AZURE_STORAGE_TABLE_URL.startswith("https://") and 
                (".table.core.windows.net" in AZURE_STORAGE_TABLE_URL or
                 ".table.cosmos." in AZURE_STORAGE_TABLE_URL)):
            warnings.append(f"AZURE_STORAGE_TABLE_URL format may be incorrect: {AZURE_STORAGE_TABLE_URL}")
            warnings.append("Expected format: https://<account-name>.table.core.windows.net")
    
    # Print warnings
    for warning in warnings:
        print(f"WARNING: {warning}")
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")