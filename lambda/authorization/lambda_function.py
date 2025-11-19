import json
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

VALID_TOKEN = os.environ.get('AUTH_TOKEN')

def lambda_handler(event, context):
    try:
        logger.info("Event incoming, trying to authorize request")
        logger.info(f"Full event: {json.dumps(event)}")
        
        # HTTP APIs lowercase all headers
        headers = event.get("headers", {})
        token = headers.get("authorization") or headers.get("Authorization")
        
        logger.info(f"Token received: {token}")
        logger.info(f"Expected token: {VALID_TOKEN}")
        
        # Strip "Bearer " prefix if present
        if token:
            if token.startswith("Bearer "):
                token = token[7:]  # Remove "Bearer "
            elif token.startswith("bearer "):
                token = token[7:]  # Remove "bearer "
        
        logger.info(f"Token after strip: {token}")
        
        if token and token == VALID_TOKEN:
            logger.info("Authorization successful")
            return {
                "isAuthorized": True,
                "context": {
                    "user": "authorized"
                }
            }
        
        logger.info("Authorization failed")
        return {
            "isAuthorized": False
        }
        
    except Exception as e:
        logger.error(f"Error in authorizer: {e}")
        return {
            "isAuthorized": False
        }