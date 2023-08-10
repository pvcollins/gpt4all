# https://medium.com/dataherald/how-to-connect-llm-to-sql-database-with-llamaindex-fae0e54de97c

from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain import SQLDatabase  #, SQLDatabaseChain

from sqlalchemy.sql.elements import quoted_name
from sqlalchemy import create_engine, MetaData
from llama_index import LLMPredictor, ServiceContext, SQLDatabase, VectorStoreIndex
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from langchain import OpenAI
import openai

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
os.environ['OPENAI_API_KEY'] = config['openAI_key']
openai.api_key = config['openAI_key']

# 2. Connect to the database --------------------------------------------------------------------------------------
engine = create_engine('mssql+pyodbc://PVC-LAPTOP/PortfolioDb?driver=SQL+Server',
                       connect_args={'timeout': 60})

# tables_restricted_list = []
tables_restricted_list = ["AssetClassMap", "Instrument", "Position"]

# load all table definitions
metadata_obj = MetaData()
if tables_restricted_list:
    metadata_obj.reflect(engine, only=tables_restricted_list)
    metadata_obj.reflect(engine, schema="dbo", only=tables_restricted_list)  # TIMEOUT HERE if not set above!
else:
    metadata_obj.reflect(engine, schema="dbo")
print("Meta data extracted")

sql_database = SQLDatabase(engine)

table_node_mapping = SQLTableNodeMapping(sql_database)

table_schema_objs = []
for table_name in metadata_obj.tables.keys():
    table_schema_objs.append(SQLTableSchema(table_name=table_name))

# We dump the table schema information into a vector index. The vector index is stored within the context builder for future use.

# NB this needs openAI!!!
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
print("Created vector store")

# 3. Setup LLM --------------------------------------------------------------------------------------------------------
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

base_model = LlamaForCausalLM.from_pretrained(
#   "vicgalle/gpt-j-6B-alpaca-gpt4",  # GPTJ model
    "chavinlo/alpaca-native",         # Llama model
    load_in_8bit=True,
#    from_tf=True,
    device_map='auto',
    cache_dir=os.environ['HUGGINGFACE_HUB_CACHE'],
    resume_download=True,
    offload_folder=os.environ['PC_HUGGINGFACE_OFFLOAD'],
    quantization_config=quantization_config
)

# tokenizer = LlamaTokenizer.from_pretrained("vicgalle/gpt-j-6B-alpaca-gpt4") # GPTJ model
tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")          # Llama model
print("No messages 2")

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=512,  # ------- PVC Larger for Db query ------------------------
#    padding=512,
    temperature=0,
    top_p=0.95,
    top_k=1,
    repetition_penalty=1.2
)
print("xformer messages 3")

local_llm = HuggingFacePipeline(pipeline=pipe)

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
# ---------------------------------------------------------------------------------------

llm_predictor = LLMPredictor(llm=local_llm)
#llm_predictor = LLMPredictor(llm=local_llm, prompt=PROMPT)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

# 4. Create the query engine ---------------------------------------------------------------------------------------
# We construct a SQLTableRetrieverQueryEngine.
# Note that we pass in the ObjectRetriever so that we can dynamically retrieve the table during query-time.
# ObjectRetriever: A retriever that retrieves a set of query engine tools.
query_engine = SQLTableRetrieverQueryEngine(
    sql_database,
    obj_index.as_retriever(similarity_top_k=1),
    text_to_sql_prompt=PROMPT,
    service_context=service_context,
)

# 5. Ask a query ---------------------------------------------------------------------------------------------
response = query_engine.query("How many instruments in the instrument table?")

print(response)
print(response.metadata['sql_query'])
print(response.metadata['result'])

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

Question: {query_str}"""

PROMPT = PromptTemplate(
    input_variables=["query_str", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

#db_chain = SQLDatabaseChain.from_llm(local_llm, db, prompt=PROMPT, verbose=True,
#                                     use_query_checker=True,
##                                    return_intermediate_steps=True,
#                                     top_k=3)
#result = db_chain.run(dict(input="How many instruments in the foobar table?",
#                           table_info=db.get_table_info(),
#                           dialect=db.dialect))

print("Fin")
