import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from dotenv import load_dotenv
load_dotenv()

from app.agents.deep_research_agent.tools.tavily_search import search_prioritized

query = "Plantozyme application timing banana yield enhancement dosage per litre India"
results = search_prioritized(query, max_results=4)

for r in results:
    print(f"URL: {r['url']}")
    print(f"Content: {r['content'][:300]}...\n")
