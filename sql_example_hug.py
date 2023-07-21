# ---------------------------------------------------------------------------------------------------------------
# Example connection to SQL database from langchain
# NOTE: 1) That this only works for one Schema at a time
#       2) The sqlalchemy/engine/base.py (version 2.0.15) was modified in row ~1970 to fix the passing parameters
#
# P Collins (copied and modified from internet example)  10th June 2023
# ---------------------------------------------------------------------------------------------------------------

from langchain import SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
# from langchain import PromptTemplate
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.chains import LLMChain

from sqlalchemy.sql.elements import quoted_name

#from langchain.agents import create_sql_agent
#from langchain.agents.agent_toolkits import SQLDatabaseToolkit
#from langchain.agents import AgentExecutor

# For testing -----------------------
from sqlalchemy import create_engine, text  # Let's try this
import pandas as pd
# -----------------------------------
import json
import os
from dotenv import load_dotenv
load_dotenv()

gblConfig = 'C:\\PVC\\config.json'

# Step 2: Load the config.json into a dictionary for use
with open(gblConfig) as config_file:
    config = json.load(config_file)

# set up API key
os.environ['OPENAI_API_KEY'] = config['rdwl_openAI_key']

# TEST ----------------------------------------------------------------------
# engine = create_engine(url='mssql+pyodbc://PVC-LAPTOP/PortfolioDb?driver=SQL Server', connect_args={'timeout': 60})
# sql = r"SELECT * FROM PortfolioDb.[dbo].[AssetClassMap] (nolock);"
# df = pd.read_sql_query(sql=sql, con=engine)
# engine.dispose()
# END -----------------------------------------------------------------------

db = SQLDatabase.from_uri('mssql+pyodbc://PVC-LAPTOP/PortfolioDb?driver=SQL+Server',
#                          include_tables=['Instrument', 'AssetClassMap', 'Trade'],  # Subset of the Tables/Views
#                          schema=None,  # s/b quoted_name 'dbo',
                          schema=quoted_name("dbo", True),  # ONLY one Schema at a time!!
                          sample_rows_in_table_info=3,
                          max_string_length=500,
                          view_support=False,  # Include Views as well as Table
                          indexes_in_table_info=False,
                          engine_args={'connect_args': {'timeout': 60, 'use_setinputsizes': False}})


# llm = OpenAI(temperature=0, verbose=True)

# This llm needs an internet link
#llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 1e-10})
# Hopefully this llm doesn't but uses the cached version
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer
model_name = "google-flan-t5-xl"
model_name = "google-flan-t5-base"
model_name = "google/flan-t5-base"
model_name = "google/flan-t5-small"
# model_name = "bert-base-cased"
# model_name = "bert-base-uncased"

llm = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text2text-generation",
    model_kwargs={"temperature": 0, "max_length": 4096},
)

# sanitycheck_model_names = ['bert-base-uncased', 'bert-base-cased']

# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# llm = T5ForConditionalGeneration.from_pretrained(model_name)
# model = T5PreTrainedModel.from_pretrained(model_name)
# model = T5PreTrainedModel.from_pretrained(model_name, from_tf=True)
# model = T5PreTrainedModel.from_pretrained(model_name, subfolder=os.environ['TRANSFORMERS_CACHE'])
# model = AutoModel.from_pretrained(model_name)

#for model_ in sanitycheck_model_names:
#    print("\nModel: " + model_)
#    try:
#        model = AutoModel.from_pretrained(model_)
#        print("Model successfully loaded!")
#        del model
#    except:
#        pass


# --------------------------------------------------------------------
template = """Question: {question}

Answer: Let's think step by step."""
PROMPT = PromptTemplate(template=template, input_variables=["question"])

question = "When was Google founded?"

tstChain = LLMChain(llm=llm, prompt=PROMPT, verbose=False)
chainOutput = tstChain({'question': question})['text']  # HERE
print(chainOutput)

#chainOutput = tstChain.predict(question)  # CRASH
#print(chainOutput)

print(LLMChain.run(llm=llm, prompt=question))  # CRASH
print(LLMChain.run(question))
# =--------------------------------------------------------------

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

Question: {input}"""

PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)

# TOP --------------------------------------------------------------------
# A)
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
# db_chain.run("How many instruments are there in the foobar table?")
db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True)
db_chain.run(dict(input="How many instruments in the foobar table?", table_info=db.get_table_info(), dialect=db.dialect))

# or, B)

db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)
db_chain.run(dict(input="How many assetclass=equity are there?", table_info=db.get_table_info(), dialect=db.dialect))
print('fin')
