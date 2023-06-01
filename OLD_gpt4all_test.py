"""GPT4All from
https://artificialcorner.com/gpt4all-is-the-local-chatgpt-for-your-documents-and-it-is-free-df1016bc335
"""
# Version 1
# pip install pygpt4all
# pip install langchain==0.0.149
# pip install unstructured
# pip install pdf2image
# pip install pytesseract
# pip install pypdf
# pip install faiss-cpu

# Version 2
# pip install pygpt4all==1.0.1
# pip install pyllamacpp==1.0.6
# pip install langchain==0.0.149
# pip install unstructured==0.6.5
# pip install pdf2image==1.16.3
# pip install pytesseract==0.3.10
# pip install pypdf==3.8.1
# pip install faiss-cpu==1.7.4


# pip install https://mirrorservice.org/sites/sourceware.org/pub/releases/gcc-13
# install minGW Windows C Compiler instead of line above

# Version 3 instead of pygpt4all
# pip install gpt4all

# ------------------- THIS WORKS IN DEBUG MODE ONLY!!! ------------------

from gpt4all import GPT4All
import json

with open('config.json') as config_file:
    config = json.load(config_file)

model_path = config['model_path']
model = config['model_name']

model = GPT4All(model_name=model, model_path=model_path, allow_download=True)
model.generate("Name 3 colors, ", streaming=True)

ans = model.generate("Name 3 colors, ", streaming=False)
print(f'The answer is: {ans}')

messages = [{"role": "user", "content": "Name 3 colors"}]
model.chat_completion(messages)

print("Fin")
