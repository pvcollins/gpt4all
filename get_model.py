import os
import json
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import hf_hub_download, snapshot_download



gblConfig = 'C:\\PVC\\config.json'

# Step 2: Load the config.json into a dictionary for use
with open(gblConfig) as config_file:
    config = json.load(config_file)

# set up API key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = config['hugging_face_key']
HUGGING_FACE_API_KEY = config['hugging_face_key']

# Replace this if you want to use a different model
# model_id, filename = ("google/flan-t5-small", "pytorch_model.bin")
model_id, filename = ("chavinlo/alpaca-native", "pytorch_model.bin")

downloaded_model_path = hf_hub_download(
    repo_id=model_id,
    filename=filename,
    token=HUGGING_FACE_API_KEY
)

print(downloaded_model_path)

hf_hub_download(repo_id="facebook/mbart-large-50", filename="config.json")
hf_hub_download(repo_id="google/flan-t5-xl", filename="config.json")
# snapshot_download(repo_id="google/flan-t5-base")
# snapshot_download(repo_id="google/flan-t5-small")
# snapshot_download(repo_id="google/flan-t5-xl")
snapshot_download(repo_id="chavinlo/alpaca-native")

print('fin')
