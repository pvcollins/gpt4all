from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

# from sqlalchemy import create_engine  # Let's try this

from dotenv import load_dotenv
load_dotenv()

# TOP ----------------------------------------------------------------------
# a)
# engine = create_engine(r'mssql+pyodbc://PETERCOL\CLEANER/PortfolioDb?driver=SQL Server')
# db = engine.url

# or, B)
db = SQLDatabase.from_uri(r'mssql+pyodbc://PETERCOL\CLEANER/PortfolioDb?driver=SQL Server',
                          include_tables=['Instruments', 'AssetClassMap', 'Trade']
                          )
# END --------------------------------------------------------------------

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

db_chain.run("How many instruments are there in the foobar table?")

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)
db_chain.run("How many albums by Aerosmith?")
