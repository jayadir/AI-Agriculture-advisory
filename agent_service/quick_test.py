"""
Quick Agent Test - Single Query
================================
Fast test for a single query without database persistence.

Usage:
    python quick_test.py
    python quick_test.py "Your question here"
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.agents.react_agent_v2.graph import chat_with_agent
from datetime import datetime


def quick_test(query: str):
    """Test a single query"""
    print("=" * 70)
    print("QUICK AGENT TEST")
    print("=" * 70)
    print(f"\nQuery: {query}\n")
    print("Processing...")
    
    start_time = datetime.now()
    
    try:
        # Call agent without chat history
        response_data = chat_with_agent(
            user_id="+1234567890",
            query=query,
            chat_history=[]
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\nResponse time: {elapsed:.2f}s")
        print("\n" + "=" * 70)
        print("RESPONSE:")
        print("=" * 70)
        print(response_data.get("response", "No response"))
        print("=" * 70)
        
        # Show thread ID for reference
        thread_id = response_data.get("thread_id")
        if thread_id:
            print(f"\nThread ID: {thread_id}")
        
        return response_data
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default test query
        query = "What are the common diseases in wheat crops?"
    
    quick_test(query)
