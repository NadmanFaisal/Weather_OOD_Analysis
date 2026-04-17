import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv('NUSCENES_USERNAME')

print(username)
