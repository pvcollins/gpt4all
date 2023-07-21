import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

import json
import os
from dotenv import load_dotenv
load_dotenv()

gblConfig = 'C:\\PVC\\config.json'

# Step 2: Load the config.json into a dictionary for use
with open(gblConfig) as config_file:
    config = json.load(config_file)

# set up API key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = config['hugging_face_key']
os.environ['OPENAI_API_KEY'] = config['rdwl_openAI_key']

base_model = LlamaForCausalLM.from_pretrained(
    "chavinlo/alpaca-native",
#    load_in_8bit=True,
#    from_tf=True,
    device_map='auto',
    cache_dir=os.environ['HUGGINGFACE_HUB_CACHE'],
    resume_download=True,
    offload_folder=os.environ['PC_HUGGINGFACE_OFFLOAD']
)

tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=500,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)
llm_chain = LLMChain(prompt=prompt, llm=local_llm)


template = """
Write a SQL Query given the table name {Table} and columns as a list {Columns} for the given question : 
{question}.
"""

prompt = PromptTemplate(template=template, input_variables=["Table","question","Columns"])


def get_llm_response(tble, question, cols):
    llm_chain = LLMChain(prompt=prompt,
                         llm=local_llm
                         )
    response = llm_chain.run({"Table": tble, "question": question, "Columns": cols})
    print(response)


tble = "employee"
cols = ["id", "name", "date_of_birth", "band", "manager_id"]
question = "Query the count of employees in band L6 with 239045 as the manager ID"
get_llm_response(tble, question, cols)
