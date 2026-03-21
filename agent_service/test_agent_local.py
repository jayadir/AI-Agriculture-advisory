"""
Local Agent Testing Script
==========================
Test the complete agent flow without AWS/Twilio dependencies.

Usage:
    python test_agent_local.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone
from bson import ObjectId

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.db.mongodb import get_database
from app.models.user import UserInDB
from app.models.chat import Message
from app.agents.react_agent_v2.graph import chat_with_agent
from app.agents.react_agent_v2.learning import learn_from_session


class LocalAgentTester:
    def __init__(self):
        self.db = None
        self.test_phone = "+1234567890"
        
    async def initialize(self):
        """Initialize database connection"""
        print("=" * 70)
        print("LOCAL AGENT TESTER")
        print("=" * 70)
        print("\nInitializing database connection...")
        self.db = await get_database()
        print("Database connected")
        
    async def ensure_test_user(self):
        """Ensure test user exists in database"""
        user = await self.db["users"].find_one({"phone_number": self.test_phone})
        
        if not user:
            print(f"\nCreating test user: {self.test_phone}")
            new_user = UserInDB(
                phone_number=self.test_phone,
                full_name="Test Farmer"
            )
            await self.db["users"].insert_one(new_user.model_dump(by_alias=True))
            print("Test user created")
        else:
            print(f"\nTest user exists: {self.test_phone}")
            
    async def get_chat_context(self):
        """Get existing chat session if available"""
        existing_session = await self.db["chat_sessions"].find_one(
            {"user_phone": self.test_phone},
            sort=[("updated_at", -1)]
        )
        
        thread_id = None
        chat_history = []
        
        if existing_session:
            thread_id = existing_session.get("thread_id")
            messages = existing_session.get("messages", [])
            chat_history = messages[-10:] if len(messages) > 10 else messages
            print(f"\nContinuing existing session (Thread ID: {thread_id})")
            print(f"Chat history: {len(chat_history)} messages")
        else:
            print("\nStarting new chat session")
            
        return thread_id, chat_history
    
    async def save_chat_session(self, thread_id: str, query: str, response: str):
        """Save or update chat session"""
        try:
            user_msg = Message(
                role="user",
                content=query,
                timestamp=datetime.now(timezone.utc)
            )
            assistant_msg = Message(
                role="assistant",
                content=response,
                timestamp=datetime.now(timezone.utc)
            )
            
            existing_session = await self.db["chat_sessions"].find_one(
                {"user_phone": self.test_phone},
                sort=[("updated_at", -1)]
            )
            
            if existing_session and existing_session.get("thread_id") == thread_id:
                # Update existing session
                await self.db["chat_sessions"].update_one(
                    {"_id": existing_session["_id"]},
                    {
                        "$push": {
                            "messages": {
                                "$each": [
                                    user_msg.model_dump(),
                                    assistant_msg.model_dump()
                                ]
                            }
                        },
                        "$set": {
                            "updated_at": datetime.now(timezone.utc),
                            "summary": query[:50]
                        }
                    }
                )
                print("\nChat session updated")
            else:
                # Create new session
                session_dict = {
                    "_id": ObjectId(),
                    "user_phone": self.test_phone,
                    "thread_id": thread_id,
                    "messages": [
                        user_msg.model_dump(),
                        assistant_msg.model_dump()
                    ],
                    "summary": query[:50],
                    "updated_at": datetime.now(timezone.utc)
                }
                
                await self.db["chat_sessions"].insert_one(session_dict)
                print("\nNew chat session created")
                
        except Exception as e:
            print(f"\nError saving chat session: {e}")
    
    async def process_query(self, query: str):
        """Process a single query through the agent"""
        print("\n" + "=" * 70)
        print(f"USER QUERY: {query}")
        print("=" * 70)
        
        # Get chat context
        thread_id, chat_history = await self.get_chat_context()
        
        # Call agent
        print("\nProcessing with agent...")
        start_time = datetime.now()
        
        try:
            response_data = chat_with_agent(
                self.test_phone,
                query,
                chat_history
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            thread_id = response_data.get("thread_id")
            response_text = response_data.get("response", "")
            
            print(f"\nAgent response time: {elapsed:.2f}s")
            print("\n" + "-" * 70)
            print("AGENT RESPONSE:")
            print("-" * 70)
            print(response_text)
            print("-" * 70)
            
            # Save to database
            await self.save_chat_session(thread_id, query, response_text)
            
            # Trigger background learning
            if thread_id:
                print("\nTriggering background learning...")
                asyncio.create_task(learn_from_session(thread_id))
            
            return response_data
            
        except Exception as e:
            print(f"\nError processing query: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def interactive_mode(self):
        """Interactive chat mode"""
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("Type your queries (or 'quit' to exit, 'history' to view chat)")
        print("=" * 70 + "\n")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting...")
                    break
                    
                if query.lower() == 'history':
                    await self.show_history()
                    continue
                    
                if query.lower() == 'new':
                    await self.clear_session()
                    continue
                
                await self.process_query(query)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    async def show_history(self):
        """Show chat history"""
        session = await self.db["chat_sessions"].find_one(
            {"user_phone": self.test_phone},
            sort=[("updated_at", -1)]
        )
        
        if not session:
            print("\nNo chat history found")
            return
            
        print("\n" + "=" * 70)
        print(f"CHAT HISTORY (Thread: {session.get('thread_id', 'N/A')})")
        print("=" * 70)
        
        messages = session.get("messages", [])
        for i, msg in enumerate(messages[-20:], 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            
            prefix = "USER" if role == "user" else "AGENT"
            print(f"\n[{i}] {prefix}: {content[:200]}")
            if len(content) > 200:
                print("    ...")
        
        print("\n" + "=" * 70)
    
    async def clear_session(self):
        """Clear current chat session"""
        result = await self.db["chat_sessions"].delete_many(
            {"user_phone": self.test_phone}
        )
        print(f"\nCleared {result.deleted_count} session(s)")
    
    async def run_test_queries(self, queries):
        """Run a list of test queries"""
        print("\n" + "=" * 70)
        print("RUNNING TEST QUERIES")
        print("=" * 70)
        
        for i, query in enumerate(queries, 1):
            print(f"\n\n--- Test Query {i}/{len(queries)} ---")
            await self.process_query(query)
            
            if i < len(queries):
                print("\nWaiting 2 seconds before next query...")
                await asyncio.sleep(2)
        
        print("\n" + "=" * 70)
        print("ALL TEST QUERIES COMPLETED")
        print("=" * 70)


async def main():
    tester = LocalAgentTester()
    await tester.initialize()
    await tester.ensure_test_user()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Run predefined test queries
            test_queries = [
                "What are the symptoms of wheat rust?",
                "How do I treat it?",
                "What about fertilizer recommendations?",
            ]
            await tester.run_test_queries(test_queries)
        elif sys.argv[1] == "--history":
            await tester.show_history()
        elif sys.argv[1] == "--clear":
            await tester.clear_session()
        elif sys.argv[1] == "--query":
            if len(sys.argv) > 2:
                query = " ".join(sys.argv[2:])
                await tester.process_query(query)
            else:
                print("Usage: python test_agent_local.py --query <your question>")
        else:
            print("Unknown option. Use: --test, --history, --clear, --query, or no args for interactive mode")
    else:
        # Interactive mode
        await tester.interactive_mode()


if __name__ == "__main__":
    print("\nStarting Local Agent Tester...\n")
    asyncio.run(main())
