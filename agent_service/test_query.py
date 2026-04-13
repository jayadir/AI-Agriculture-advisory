import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from app.agents.deep_research_agent.graph import chat_with_agent

res = chat_with_agent(
    user_id="test", 
    query=" I am a small farmer from Tamil Nadu growing paddy on 2 acres. Am I eligible for PM-KISAN and how much money will I receive per year? Also how can I check my application status?", 
    chat_history=[]
)

print(res['response'])
