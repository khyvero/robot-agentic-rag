import json
import os

def load_config(filename="config/config.json"):
    # reads the json configuration file
    if not os.path.exists(filename):
        print(f"Error: '{filename}' not found.")
        print("Please ensure it is in the same folder.")
        quit()

    with open(filename, 'r') as f:
        print(f"> Loaded configuration from {filename}")
        return json.load(f)