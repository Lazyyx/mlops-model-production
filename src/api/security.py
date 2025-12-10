import os
from fastapi import Header, HTTPException, Depends

# for now we store in env file
API_SECRET_TOKEN = os.environ.get("APP_TOKEN") 

async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    
    # si le secret n'est pas défini, on bloque en production
    if not API_SECRET_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Erreur de configuration du serveur: Le secret APP_TOKEN est manquant."
        )

    # token manquant ou ne correspond pas au secret
    if x_api_key is None or x_api_key != API_SECRET_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Accès refusé. Token 'X-API-Key' invalide ou manquant."
        )