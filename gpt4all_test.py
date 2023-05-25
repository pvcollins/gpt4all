"""GPT4All from
https://artificialcorner.com/gpt4all-is-the-local-chatgpt-for-your-documents-and-it-is-free-df1016bc335
"""

# pip install pygpt4all
# pip install langchain==0.0.149
# pip install unstructured
# pip install pdf2image
# pip install pytesseract
# pip install pypdf
# pip install faiss-cpu

# pip install https://mirrorservice.org/sites/sourceware.org/pub/releases/gcc-13
# install minGW Windows C Compiler instead of line above

from pygpt4all.models.gpt4all import GPT4All


def new_text_callback(text):
    print(text, end="")


model = GPT4All('C:/Dropbox/IT_Stuff/Python3.11/gpt4all/models/gpt4all-converted.bin', n_parts=-1)
model = GPT4All('./models/gpt4all-converted.bin')
model.generate("Once upon a time, ", n_predict=55, new_text_callback=new_text_callback)
