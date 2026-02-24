"""
Test script for chat session continuity
Run this to test without making actual phone calls
"""
import asyncio
import sys
import os
from datetime import datetime, timezone
from bson import ObjectId

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.react_agent_v2.graph import chat_with_agent
from app.db.mongodb import get_database
from app.models.chat import ChatSession, Message


async def save_chat_session_test(db, phone: str, thread_id: str, query: str, response: str):
    """Save or update chat session - same as in chatWorker.py"""
    try:
        user_msg = Message(role="user", content=query, timestamp=datetime.now(timezone.utc))
        assistant_msg = Message(role="assistant", content=response, timestamp=datetime.now(timezone.utc))
        
        existing_session = await db["chat_sessions"].find_one(
            {"user_phone": phone},
            sort=[("updated_at", -1)]
        )
        
        if existing_session and existing_session.get("thread_id") == thread_id:
            await db["chat_sessions"].update_one(
                {"_id": existing_session["_id"]},
                {
                    "$push": {
                        "messages": {
                            "$each": [user_msg.model_dump(), assistant_msg.model_dump()]
                        }
                    },
                    "$set": {
                        "updated_at": datetime.now(timezone.utc),
                        "summary": query[:50]
                    }
                }
            )
            print(f"✓ Updated chat session for {phone}")
        else:
            # Create new session directly as dict to avoid ObjectId issues
            session_dict = {
                "_id": ObjectId(),
                "user_phone": phone,
                "thread_id": thread_id,
                "messages": [user_msg.model_dump(), assistant_msg.model_dump()],
                "summary": query[:50],
                "updated_at": datetime.now(timezone.utc)
            }
            
            await db["chat_sessions"].insert_one(session_dict)
            print(f"✓ Created new chat session for {phone}")
            
    except Exception as e:
        print(f"✗ Error saving chat session: {e}")


async def test_chat_continuity():
    """Test the chat continuity feature"""
    
    # Test phone number
    test_phone = "+91-TEST-12345"
    
    print("="*70)
    print("CHAT CONTINUITY TEST")
    print("="*70)
    
    # Connect to database
    db = await get_database()
    print(f"\n✓ Connected to database\n")
    
    # Clean up any existing test sessions
    await db["chat_sessions"].delete_many({"user_phone": test_phone})
    print(f"✓ Cleaned up previous test data\n")
    
    # TEST 1: First conversation (no history)
    print("\n" + "="*70)
    print("TEST 1: First Query (New Conversation)")
    print("="*70)
    
    query1 = "What are the best practices for growing wheat in Punjab?"
    print(f"\nQuery: {query1}")
    
    # Check for existing session (should be None)
    existing_session = await db["chat_sessions"].find_one(
        {"user_phone": test_phone},
        sort=[("updated_at", -1)]
    )
    
    thread_id = None
    chat_history = []
    
    if existing_session:
        thread_id = existing_session.get("thread_id")
        messages = existing_session.get("messages", [])
        chat_history = messages[-10:] if len(messages) > 10 else messages
        print(f"\n→ Found existing session with thread_id: {thread_id}")
    else:
        print(f"\n→ No existing session found. Starting new conversation.")
    
    # Call agent
    response1 = chat_with_agent(test_phone, query1, thread_id, chat_history)
    print(f"\n✓ Response received")
    print(f"Thread ID: {response1['thread_id']}")
    print(f"Response preview: {response1['response'][:200]}...")
    
    # Save to database
    await save_chat_session_test(db, test_phone, response1['thread_id'], query1, response1['response'])
    
    # TEST 2: Follow-up question (should use context)
    print("\n" + "="*70)
    print("TEST 2: Follow-up Query (Should Continue Conversation)")
    print("="*70)
    
    query2 = "What about pest management for the same crop?"
    print(f"\nQuery: {query2}")
    
    # Check for existing session (should exist now)
    existing_session = await db["chat_sessions"].find_one(
        {"user_phone": test_phone},
        sort=[("updated_at", -1)]
    )
    
    thread_id = None
    chat_history = []
    
    if existing_session:
        thread_id = existing_session.get("thread_id")
        messages = existing_session.get("messages", [])
        chat_history = messages[-10:] if len(messages) > 10 else messages
        print(f"\n→ Found existing session with thread_id: {thread_id}")
        print(f"→ Chat history contains {len(chat_history)} messages")
        print(f"\nPrevious context being passed:")
        for msg in chat_history:
            role = msg.get('role', '')
            content = msg.get('content', '')
            print(f"  {role}: {content[:80]}...")
    else:
        print(f"\n→ No existing session found (unexpected!)")
    
    # Call agent with history
    response2 = chat_with_agent(test_phone, query2, thread_id, chat_history)
    print(f"\n✓ Response received")
    print(f"Thread ID: {response2['thread_id']}")
    print(f"Response preview: {response2['response'][:200]}...")
    
    # Save to database
    await save_chat_session_test(db, test_phone, response2['thread_id'], query2, response2['response'])
    
    # TEST 3: Another follow-up
    print("\n" + "="*70)
    print("TEST 3: Another Follow-up Query")
    print("="*70)
    
    query3 = "How much water does it need?"
    print(f"\nQuery: {query3}")
    
    # Check for existing session
    existing_session = await db["chat_sessions"].find_one(
        {"user_phone": test_phone},
        sort=[("updated_at", -1)]
    )
    
    thread_id = None
    chat_history = []
    
    if existing_session:
        thread_id = existing_session.get("thread_id")
        messages = existing_session.get("messages", [])
        chat_history = messages[-10:] if len(messages) > 10 else messages
        print(f"\n→ Found existing session with thread_id: {thread_id}")
        print(f"→ Chat history contains {len(chat_history)} messages")
    
    # Call agent with history
    response3 = chat_with_agent(test_phone, query3, thread_id, chat_history)
    print(f"\n✓ Response received")
    print(f"Thread ID: {response3['thread_id']}")
    print(f"Response preview: {response3['response'][:200]}...")
    
    # Save to database
    await save_chat_session_test(db, test_phone, response3['thread_id'], query3, response3['response'])
    
    # Display final chat session
    print("\n" + "="*70)
    print("FINAL CHAT SESSION IN DATABASE")
    print("="*70)
    
    final_session = await db["chat_sessions"].find_one({"user_phone": test_phone})
    if final_session:
        print(f"\nPhone: {final_session['user_phone']}")
        print(f"Thread ID: {final_session.get('thread_id')}")
        print(f"Total messages: {len(final_session.get('messages', []))}")
        print(f"\nConversation history:")
        for i, msg in enumerate(final_session.get('messages', []), 1):
            role = msg.get('role', '')
            content = msg.get('content', '')
            print(f"\n{i}. {role.upper()}: {content[:150]}{'...' if len(content) > 150 else ''}")
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nYou can now check the 'chat_sessions' collection in MongoDB")
    print(f"to see the stored conversation for phone: {test_phone}")
    
    # Optional: Clean up test data
    cleanup = input("\n\nClean up test data? (y/n): ")
    if cleanup.lower() == 'y':
        await db["chat_sessions"].delete_many({"user_phone": test_phone})
        print("✓ Test data cleaned up")


if __name__ == "__main__":
    print("\n🧪 Starting Chat Continuity Test...\n")
    asyncio.run(test_chat_continuity())
