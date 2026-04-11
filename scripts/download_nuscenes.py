# src: https://github.com/li-xl/nuscenes-download
"""
Artifact: 
nuScenes Data Ingestion & Integrity Pipeline
Methodology: Design Science Research (Cycle II: Solution Design)
Researcher(s): Hasan Zahid, Nadman Abdullah Bin Faisal, Vaibhav Puram

Purpose:
This utility handles authenticated data collection from the nuScenes API.
It ensures the environment is downloaded and extracted correctly for post-hoc 
OOD analysis on BEVFormer and BEVFusion architectures.
"""

import os
import sys
import requests
import hashlib
import tarfile
import json

from dotenv import load_dotenv
from tqdm import tqdm

# Gets the project directory's name
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import OUTPUT_DIR, REGION, FILES

load_dotenv()

username = os.getenv('NUSCENES_USERNAME')
password = os.getenv('NUSCENES_PASSWORD')

output_dir = OUTPUT_DIR
region = REGION
files = FILES

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
        print("\tLogged in!")
        try:
            token = json.loads(response.content)["AuthenticationResult"]["IdToken"]
            return token
        except KeyError:
            print("\tAuthentication failed. 'AuthenticationResult' not found in the response.")
    else:
        print("\tFailed to login. Status code:", response.status_code)

    return None

def download_files(url, save_file, md5):
    response = requests.get(url, stream=True)

    if save_file.endswith(".tgz"):
        content_type = response.headers.get('Content-Type', '')
        if content_type == 'application/x-tar':
            save_file = save_file.replace('.tgz', '.tar')
        elif content_type != 'application/octet-stream':
            print("\tUnknown content type: ", content_type)
            return save_file

    if os.path.exists(save_file):
        print("\t", save_file, " has been downloaded!")

        md5obj = hashlib.md5()
        with open(save_file, 'rb') as file:
            for chunk in file:
                md5obj.update(chunk)
        hash = md5obj.hexdigest()
        if hash != md5:
            print(f"\t{save_file} check md5 failed, downloading again.")
        else:
            print(f"\t{save_file} check md5 success")
            return save_file

    file_size = int(response.headers.get('Content-Length', 0))
    progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024,desc=save_file, ascii=True)


    md5obj = hashlib.md5()
    with open(save_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                md5obj.update(chunk)
                file.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()

    hash = md5obj.hexdigest()
    if hash != md5:
        print(save_file,"\tcheck md5 failed")
    else:
        print(save_file,"\tcheck md5 success")

    return save_file

def extract_archive(file_path):
    """
    Unpacks .tar/.tgz files into the designated dataroot.
    Adheres to the specific directory structure required by 
    the nuScenes-devkit.
    """
    original_folder = os.path.dirname(file_path)
    print(f"\tExtracting {file_path} to {original_folder}")

    try:
        with tarfile.open(file_path, 'r:*') as tar:
            tar.extractall(path=original_folder)
        print("\tExtraction successful.")
    except Exception as e:
        print(f"\tExtraction failed: {e}")

def main():

    # Authentication...
    print("Loginging...")
    bearer_token = login(username, password)
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json',
    }

    # Checks whether the output directory exists
    if not os.path.exists(output_dir):
        print(f"\tERROR: The directory '{output_dir}' was not found.")
        print("Please create the folder structure manually within the repo.")
        return

    print("Getting download urls...")
    
    # Downloads data for the specified files
    download_data = {}
    for filename, md5 in files.items():
        api_url = f'https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/v1.0/{filename}?region={region}&project=nuScenes'

        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            print(f"\t{filename} request success")
            download_url = response.json()['url']
            download_data[filename] = [download_url,os.path.join(output_dir,filename),md5]
        else:
            print(f'\trequest failed : {response.status_code}')
            print(f"\t{response.text}")

    print("Downloading files...")

    os.makedirs(output_dir,exist_ok=True)
    for output_name,(download_url,save_file,md5) in download_data.items():
        save_file = download_files(download_url,save_file,md5)
        download_data[output_name] = [download_url,save_file,md5]

    # Extracts the .tar/.tgz files into the designated folder
    print("Extracting files...")
    for output_name,(download_url,save_file,md5) in download_data.items():
        extract_archive(save_file)

if __name__ == "__main__":
    main()
