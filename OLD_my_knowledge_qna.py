# pip install llama-cpp-python

from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# function for loading only TXT files
from langchain.document_loaders import TextLoader
# text splitter for create chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# to be able to load the pdf files
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
# Vector Store Index to create our database about our knowledge
from langchain.indexes import VectorstoreIndexCreator
# LLamaCpp embeddings from the Alpaca model
from langchain.embeddings import LlamaCppEmbeddings
# FAISS  library for similarity search
from langchain.vectorstores.faiss import FAISS
import os  # for interaction with the files
import datetime
import json

with open('config.json') as config_file:
    config = json.load(config_file)

model_path = config['model_path']
model_name = config['embedding_model_name']
model = config['model_other']

# assign the path for the 2 models GPT4All and Alpaca for the embeddings
gpt4all_path = model_path + "\\" + model  # "'.\models\gpt4all-converted.bin'
llama_path = model_path + "\\" + model_name  # '.\models\ggml-model-q4_0.bin'
# Callback manager for handling the calls with  the model
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# create the embedding object
embeddings = LlamaCppEmbeddings(model_path=llama_path)
# create the GPT4All llm object
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)


# Split text
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def create_index(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return search_index


def similarity_search(query, index):
    # k is the number of similarity searched that matches the query
    # default is 4
    matched_docs = index.similarity_search(query, k=3)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources


# get the list of pdf files from the docs directory into a list  format
pdf_folder_path = './docs'
doc_list = [s for s in os.listdir(pdf_folder_path) if s.endswith('.pdf')]
num_of_docs = len(doc_list)
# create a loader for the PDFs from the path
general_start = datetime.datetime.now()  # not used now but useful
print("starting the loop...")
loop_start = datetime.datetime.now()  # not used now but useful
print("generating fist vector database and then iterate with .merge_from")

# create a loader for the PDFs from the path
loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[0]))
# load the documents with Langchain
docs = loader.load()
# Split in chunks
chunks = split_chunks(docs)
# create the db vector index
db0 = create_index(chunks)

print("Main Vector database created. Start iteration and merging...")
for i in range(1, num_of_docs):
    print(doc_list[i])
    print(f"loop position {i}")
    loader = PyPDFLoader(os.path.join(pdf_folder_path, doc_list[i]))
    start = datetime.datetime.now()  # not used now but useful
    docs = loader.load()
    chunks = split_chunks(docs)
    dbi = create_index(chunks)
    print("start merging with db0...")
    db0.merge_from(dbi)
    end = datetime.datetime.now()  # not used now but useful
    elapsed = end - start  # not used now but useful
    # total time
    print(f"completed in {elapsed}")
    print("-----------------------------------")
loop_end = datetime.datetime.now()  # not used now but useful
loop_elapsed = loop_end - loop_start  # not used now but useful
print(f"All documents processed in {loop_elapsed}")
print(f"the database is done with {num_of_docs} subset of db index")
print("-----------------------------------")
print(f"Merging completed")
print("-----------------------------------")
print("Saving Merged Database Locally")
# Save the database locally
db0.save_local("my_faiss_index")
print("-----------------------------------")
print("merged database saved as my_faiss_index")
general_end = datetime.datetime.now()  # not used now but useful
general_elapsed = general_end - general_start  # not used now but useful
print(f"All indexing completed in {general_elapsed}")
print("-----------------------------------")
