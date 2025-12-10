import os
from redis import Redis
from fastapi import Header, HTTPException, Depends


MAX_CALLS = 100
REDIS_HOST = "redis"
# for now we store in env file
API_SECRET_TOKEN = os.environ.get("APP_TOKEN") 

# init redis only once
try:
    redis_client = Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    redis_client.ping()
    print("Connexion Redis réussie.")
except Exception as e:
    print(f"Erreur de connexion Redis: {e}")
    redis_client = None

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
    
    return x_api_key

async def limit_session_calls(session_id: str = Depends(verify_api_key)):
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Le service de session est indisponible.")

    # on utilise la clé session pour faire le compteur
    key = f"calls:{session_id}" 
    
    call_count = redis_client.incr(key)

    if call_count > MAX_CALLS:
        raise HTTPException(
            status_code=429,
            detail=f"Limite de {MAX_CALLS} appels par session dépassée."
        )
    
    # expire après 1 heure (3600 secondes)
    if call_count == 1:
        redis_client.expire(key, 3600)