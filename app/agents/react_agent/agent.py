import os
import uuid
import re
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from app.services.llm import _get_model
from langchain.prompts import ChatPromptTemplate
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import ToolMessage
from langchain_tavily import TavilySearch
from langchain.schema import Document

from app.core.config import settings
from app.tools.retrieval_tool import retrieval_tool
from app.tools.web_search_tool import web_search_tool
from app.rag.embeddings import get_embedder




load_dotenv()
tavily_client=TavilySearch()  
client = MongoClient(settings.MONGODB_URI)
checkpointer=MongoDBSaver(client=client,db_name=settings.DB_NAME)
agent=None
def get_react_agent(tools):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an agriculture expert assistant. "
         "You are provided with 2 tools: one is a retriever which can fetch you documents related to pests, and the other is a web search tool. "
         "ALWAYS FIRST USE THE retrieval_tool. "
         "If you are not satisfied with the results from the retriever, then use the web search tool to get more information. "
         "Always provide references for the information you provide to the user."
        ),
        ("human", "{user_query}")
    ])
    llm=_get_model()
    global agent
    if agent is None:
        agent=create_react_agent(llm,tools,checkpointer=checkpointer)
    return agent, prompt_template

def chat_with_agent(user_id:str,query:str,thread_id:str=None):
    tools = [retrieval_tool, web_search_tool]
    agent, prompt_template = get_react_agent(tools)
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        print(f"New thread created with id: {thread_id}")
    config = {"configurable": {"thread_id": thread_id}}
    final_response=""
    for event in agent.stream(
        {"messages": prompt_template.format_messages(user_query=query)},        config=config,
        stream_mode="values"
    ):
        msg=event["messages"][-1]
        if msg.type=="ai":
            final_response=msg.content
    return {"user_id": user_id, "thread_id": thread_id, "response": final_response}

def extract_urls(text: str):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    return re.findall(url_pattern, text)

      
def learn_from_session(thread_id:str):
    print(f"--- [Background] Starting knowledge ingestion for Thread {thread_id} ---")  
    config = {"configurable": {"thread_id": thread_id}}
    if agent is None:
        print("Agent not initialized.")
        return
    state=agent.get_state(config=config)
    if not state.values:
        print(f"No interactions found for Thread {thread_id}.")
        return
    messages=state.values.get("messages",[])
    urls_to_scrape=set()
    for msg in messages:
        if isinstance(msg,ToolMessage) and msg.name=="web_search":
            urls=extract_urls(msg.content)
            for url in urls:
                if "youtube.com" not in url:
                    urls_to_scrape.add(url.rstrip(','))
    if not urls_to_scrape:
        print("[Background] No valid URLs found to scrape.")
        return
    print(f"[Background] Scraping {len(urls_to_scrape)} URLs: {urls_to_scrape}")
    
    new_docs=[]
    try:
        response=tavily_client.extract(urls=list(urls_to_scrape))
        for result in response.get("results",[]):
            content=result.get("raw_content","")
            if not content or len(content)<50:
                continue
            print(f"Extracted: {result['url'][:30]}... ({len(content)} chars)")
            
            doc = Document(
                page_content=content[:20000],
                metadata={
                    "source": result["url"],
                    "title": "Web Search Result", 
                    "ingested_at": str(datetime.utcnow()),
                    "thread_origin": thread_id,
                    "type": "tavily_extracted_content"
                }
            )
            new_docs.append(doc)
    except Exception as e:
        print(f"[Background] Tavily Extract Error: {e}")
        return
    
    if new_docs:
        try:
            embedder=get_embedder()
            
        
                    
    
    
