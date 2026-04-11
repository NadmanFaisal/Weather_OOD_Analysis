import os
import requests
import hashlib
import tarfile
import gzip
import json

from dotenv import load_dotenv

load_dotenv()

username = os.getenv('NUSCENES_USERNAME')
password = os.getenv('NUSCENES_PASSWORD')

output_dit = "../data/"
region = 'us'

download_files = {
    "v1.0-test_meta.tgz":"b0263f5c41b780a5a10ede2da99539eb",
}

def login(username, password):
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }

    data = json.dumps({
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
        "AuthParameters": {
            "USERNAME": username,
            "PASSWORD": password
        },
        "ClientMetadata": {}
    })

    response = requests.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=headers,
        data=data,
    )

    if response.status_code == 200:
        print("Logged in!")
        try:
            token = json.loads(response.content)["AuthenticationResult"]["IdToken"]
            return token
        except KeyError:
            print("Authentication failed. 'AuthenticationResult' not found in the response.")
    else:
        print("Failed to login. Status code:", response.status_code)

    return None

def main():
    print("Loginging...")

    bearer_token = login(username, password)
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json',
    }

if __name__ == "__main__":
    main()
