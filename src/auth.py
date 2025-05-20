# auth.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Request
import os
import requests
from authlib.jose import jwt, JoseError
from authlib.jose import JsonWebKey

# Azure AD configuration
AZURE_AD_TENANT_ID = os.environ.get("AZURE_TENANT_ID")
MSFT_TENANT_ID = "72f988bf-86f1-41af-91ab-2d7cd011db47"
AZURE_AD_APP_ID = os.environ.get("AZURE_AD_APP_ID")  # API app ID (audience)
AZURE_AD_AUDIENCE = f"api://{AZURE_AD_APP_ID}"
AZURE_AD_ISSUER = f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/v2.0"

AZURE_ISSUER = [
    f"https://sts.windows.net/{AZURE_AD_TENANT_ID}/", 
    f"https://login.microsoftonline.com/{MSFT_TENANT_ID}/v2.0"
]
AZURE_JWKS_URI = f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}/discovery/v2.0/keys"
AZURE_AUDIENCE = f"api://{AZURE_AD_APP_ID}"

# Bearer token parser
bearer_scheme = HTTPBearer()
JWKS = JsonWebKey.import_key_set(requests.get(AZURE_JWKS_URI).json())

# Token validation
async def validate_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """
    Validate Azure AD token
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        Claims from the token if valid
    """
    token = credentials.credentials

    try:
        
        claims = jwt.decode(
            token,
            key=JWKS,
            claims_options={
                "iss": {"values": [AZURE_ISSUER, "https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47/v2.0"]},
                # "aud": {"values": [AZURE_AUDIENCE]},
            }
        )

        # Store claims in request.state
        request.state.claims = claims

        claims.validate()
        return claims  # contains `appid`, `azp`, etc.

    except JoseError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )