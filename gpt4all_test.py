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

# ------------------- VERSION 3 --- THIS WORKS ------------------
import gpt4all
model_path = '.\models'
model = gpt4all.GPT4All(model_name='gpt4all-converted.bin', model_path=model_path, model_type='llama', allow_download=True)
model.generate("Once upon a time, ", streaming=True)

# ------------------- VERSION 1 MODIFIED ------------------
from pygpt4all.models.gpt4all import GPT4All


def new_text_callback(text):
    print(text, end="")


model = GPT4All('.\models\gpt4all-converted.bin')  # BREAKS HERE NO _ctx
model.cpp_generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)

# ------------------- ORIGINAL CODE -----------------------
from pygpt4all.models.gpt4all import GPT4All
def new_text_callback(text):
    print(text, end="")
model = GPT4All('C:\\Dropbox\\IT_Stuff\\Python3.11\\gpt4all\\models\\gpt4all-converted.bin', n_parts=-1)
model = GPT4All('./models/gpt4all-converted.bin')
model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
