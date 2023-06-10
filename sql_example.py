from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

from sqlalchemy.engine import URL
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
os.environ['OPENAI_API_KEY']  = config['rdwl_openAI_key']

# TEST ----------------------------------------------------------------------
# engine = create_engine(url='mssql+pyodbc://PVC-LAPTOP/PortfolioDb?driver=SQL Server', connect_args={'timeout': 60})
# sql = r"SELECT * FROM PortfolioDb.[dbo].[AssetClassMap] (nolock);"
# df = pd.read_sql_query(sql=sql, con=engine)
# engine.dispose()
# END -----------------------------------------------------------------------

# db = SQLDatabase.from_uri('mssql+pymssql://PVC-LAPTOP/PortfolioDb')
db = SQLDatabase.from_uri('mssql+pyodbc://PVC-LAPTOP/PortfolioDb?driver=SQL+Server',
                          include_tables=['Instrument', 'AssetClassMap', 'Trade'],
                          schema=None,  # s/b quoted_name 'dbo',
                          sample_rows_in_table_info=5,
                          max_string_length=500,
                          view_support=False,
                          indexes_in_table_info=False,
                          engine_args={'connect_args': {'timeout': 60, 'use_setinputsizes': False}})

llm = OpenAI(temperature=0, verbose=True)

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

If someone asks for the table foobar, they really mean the instrument table.

Question: {input}"""

# TOP --------------------------------------------------------------------
# A)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
# or, B)
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)
db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True)
# END --------------------------------------------------------------------

db_chain.run("How many instruments are there in the instrument table?")

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)
db_chain.run("How many assetclass=equity are there?")
