import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import AzureSearch
from typing import Annotated, Any, AsyncIterator, Optional, Sequence, Union
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, END
from urllib.parse import quote_plus 
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from langchain.agents import AgentExecutor

from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

from its_a_rag import ingestion

from langgraph.graph.message import MessagesState

class AgentState(MessagesState):
    input: str
    output: str
    state: str

llm = AzureChatOpenAI(api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint =  os.getenv("AZURE_OPENAI_ENDPOINT"),
                    model= os.getenv("AZURE_OPENAI_MODEL"),                    
                    streaming=True)

async def start_agent(state:AgentState):
    global llm
    start_agent_llm = llm
    prompt = PromptTemplate.from_template("""
    You are an agent that needs analyze the user question. \n
    Question : {input} \n
    if the question is related to stock prices answer with "stock". \n
    if the question is related to information about financial results answer with "rag". \n
    if the question is unclear or you cannot decide answer with "rag". \n
    only answer with one of the word provided.
    Your answer (stock/rag):
    """)
    chain = prompt | start_agent_llm
    input=state["messages"][-1].content
    response = await chain.ainvoke({"input": input})
    decision = response.content.strip().lower()
    return {"output":decision, "state": decision, "input": input, "messages": [response]}

async def stock_agent(state: AgentState):    
    # Import the LLM (you can use "global" to import the LLM in previous step to avoid re-creating the LLM objects)
    global llm    

    system_prompt_SQL = """
        You are a helpful AI assistant expert in querying SQL Database to find answers to user's question about stock prices. \n
        If you can't find the answer, say 'I am unable to find the answer.'
        """

    # Create the SQL Database Object and the SQL Database Toolkit Object to be used by the agent.
    engine = create_engine(
        f"mssql+pymssql://{os.getenv('SQL_USERNAME')}:{os.getenv('SQL_PWD')}@{os.getenv('SQL_SERVER')}:1433/{os.getenv('SQL_DB')}")

    db = SQLDatabase(engine=engine)
    stock_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # Create the agent using the Langhcain SQL Agent Class (create_sql_agent)
    stock_agent = create_sql_agent(llm=llm,
                                   toolkit=stock_toolkit,
                                   agent_type="openai-tools",
                                   agent_name="StockAgent",
                                   agent_description="This agent is an expert in querying SQL Database to find answers to user's question about stock prices.",
                                   agent_version="0.1",
                                   agent_author="itsarag",
                                   verbose=True,
                                   agent_executor_kwargs=dict(handle_parsing_errors=True, return_intermediate_steps=False))    
    # Structure the final prompt from the ChatPromptTemplate
    sql_prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt_SQL),
         ("user", "{question}\n ai:")])

    # Prepare the response using the invoke method of the agent
    response = await stock_agent.ainvoke(sql_prompt.format(
        question=state["input"]))
        
    # Return the response for the next agent (output and input required coming fron the Agent State)
    return {"output": response, "input": state["input"],"state":"end","messages":[response["output"]]}

async def rag_agent(state:AgentState):
    # Import the LLM (you can use "global" to import the LLM in previous step to avoid re-creating the LLM objects)
    global llm
    rag_agent_llm = llm.with_config(tags=["final_node"])
    # Define the index (use the one created in the previous challenge)
    retriever_multimodal = AzureAISearchRetriever(
        index_name=os.getenv('AZURE_SEARCH_INDEX'),
        api_key=os.getenv('AZURE_SEARCH_API_KEY'),
        service_name=os.getenv('AZURE_SEARCH_ENDPOINT'),
        top_k=5)
    # Define the chain (as it was in the previous challenge)
    chain_multimodal_rag = (
        {
            "context": retriever_multimodal | RunnableLambda(ingestion.get_image_description),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(ingestion.multimodal_prompt)
        | rag_agent_llm
        | StrOutputParser()
    )
    # prepare the response using the invoke method of the agent
    response = chain_multimodal_rag.invoke({"input": state["input"]})
    # Return the response for the next agent (output and input required coming from the Agent State)
    return {"output": response, "input": state["input"], "state":"end","messages":[ response]}

from langgraph.types import StreamMode

class MultiAgentAssistant:
    def __init__(self):

        # Create the Workflow as StateGraph using the AgentState
        workflow = StateGraph(AgentState)
        # Add the nodes (start_agent, stock_agent, rag_agent)
        workflow.add_node("start", start_agent)
        workflow.add_node("stock_agent", stock_agent)
        workflow.add_node("rag_agent", rag_agent)
        workflow.add_node("final", RunnablePassthrough(tags=["final_node"]))
        # Add the conditional edge from start -> lamba (decision) -> stock_agent or rag_agent
        workflow.add_conditional_edges(
            "start",
            lambda x: x["state"],
            {
                "stock": "stock_agent",
                "rag": "rag_agent",
                # "end": "final"
            }
        )
        # Set the workflow entry point
        workflow.set_entry_point("start")
        # Add the final edges to the END node
        workflow.add_edge("stock_agent",END)
        workflow.add_edge("rag_agent",END)
        # workflow.add_edge("final",END)
        self.runnable = workflow.compile()

    def astream(self,
                content:dict[str, Any],
                config,
                *,
                stream_mode:Optional[Union[StreamMode, list[StreamMode]]] = None,
                **kwargs:Any)->AsyncIterator[dict[str,Any]|Any]:
        return self.runnable.astream(content, config, stream_mode= stream_mode, **kwargs)

    def invoke(self,
               content:dict[str, Any],
               config,
               *,
               stream_mode:Optional[Union[StreamMode, list[StreamMode]]] = None,
               **kwargs:Any)->(dict[str,Any]|Any):
        return self.runnable.invoke(content, config, stream_mode= stream_mode, **kwargs)['output']
