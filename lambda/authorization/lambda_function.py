import os
from policies import generate_policy

VALID_TOKEN = os.environ.get('AUTH_TOKEN')

def lambda_handler(event, context):
    token = event.get("authorizationToken")

    if token == VALID_TOKEN:
        return generate_policy('user', "Allow", event["methodArn"])
    else:
        return generate_policy('user', "Deny", event["methodArn"])
        
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }