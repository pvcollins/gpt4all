# import torch
from transformers import pipeline, BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaForCausalLM  # For Llama models
from transformers import GPT2Tokenizer, GPTJForCausalLM, GPTJPreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM  # For Llama models
from langchain.llms import HuggingFacePipeline  # langchain Version 0.0.191
from langchain import PromptTemplate, LLMChain
from langchain import SQLDatabase, SQLDatabaseChain

from sqlalchemy.sql.elements import quoted_name


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
# os.environ['OPENAI_API_KEY'] = config['rdwl_openAI_key']

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

base_model = GPTJForCausalLM.from_pretrained(
    "vicgalle/gpt-j-6B-alpaca-gpt4",
#    "chavinlo/alpaca-native",
    load_in_8bit=True,
#    from_tf=True,
    device_map='auto',
    cache_dir=os.environ['HUGGINGFACE_HUB_CACHE'],
    resume_download=True,
    offload_folder=os.environ['PC_HUGGINGFACE_OFFLOAD'],
    quantization_config=quantization_config
)
print("please use 'tie_weights' method before 'infer_auto_device' function")

tokenizer = GPT2Tokenizer.from_pretrained("vicgalle/gpt-j-6B-alpaca-gpt4")
# tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")
print("No messages 2")

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=512,  # ------- PVC Larger for Db query ------------------------
#    padding=512,
    temperature=0.3,
    top_p=0.95,
    repetition_penalty=1.2
)
print("xformer messages 3")

db = SQLDatabase.from_uri('mssql+pyodbc://PVC-LAPTOP/PortfolioDb?driver=SQL+Server',
#                          include_tables=['Instrument', 'AssetClassMap', 'Trade'],  # Subset of the Tables/Views
#                          schema=None,  # s/b quoted_name 'dbo',
                          schema=quoted_name("dbo", True),  # ONLY one Schema at a time!!
                          sample_rows_in_table_info=3,
                          max_string_length=512,  # ------ PVC Larger for DB query -----------------
                          view_support=False,  # Include Views as well as Table
                          indexes_in_table_info=False,
                          engine_args={'connect_args': {'timeout': 60, 'use_setinputsizes': False}})
print("No messages 4")

local_llm = HuggingFacePipeline(pipeline=pipe)
print("No messages 5")

# -------------------------------------------------------------------------------------------------------------
template = """
Write a SQL Query given the table name {Table} and columns as a list {Columns} for the given question : 
{question}.
"""

prompt = PromptTemplate(template=template, input_variables=["Table", "question", "Columns"])
# llm_chain = LLMChain(prompt=prompt, llm=local_llm)


def get_llm_response(p_table, p_question, p_cols):
    llm_chain = LLMChain(prompt=prompt, llm=local_llm)
    response = llm_chain.run({"Table": p_table, "question": p_question, "Columns": p_cols})
    print(response)


tble = "employee"
cols = ["id", "name", "date_of_birth", "band", "manager_id"]
question = "Query the count of employees in band L6 with 239045 as the manager ID"
get_llm_response(tble, question, cols)

# -------------------------------------------------------------------------------------------------
_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

If you are unable to provide an answer, please reply as "I'm sorry, but I don't have that information at the moment.".

If someone asks for the table foobar, they really mean the instrument table.

Remember that the numbers should be formatted nicely with commas as the thousand seperator.

Question: {query}"""

PROMPT = PromptTemplate(
    input_variables=["query", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

db_chain = SQLDatabaseChain.from_llm(local_llm, db, prompt=PROMPT, verbose=True,
                                     use_query_checker=True,
#                                    return_intermediate_steps=True,
                                     top_k=3)
result = db_chain.run(dict(query="How many instruments in the foobar table?",
                           inputx="How many instruments in the foobar table?",
                           table_info=db.get_table_info(),
                           dialect=db.dialect))

print("Fin")
