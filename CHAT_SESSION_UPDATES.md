# Chat Session Management Updates

## Overview
Implemented chat session continuity for phone calls/SMS by leveraging LangGraph's built-in checkpointing mechanism. Conversations automatically maintain context across multiple interactions from the same phone number.

## Key Features Implemented

### 1. **Existing Chat Detection**
- When a call/SMS is received from a mobile number, the system checks if there's an existing chat session
- Uses MongoDB to find the most recent chat session for the phone number
- If found, continues with the existing `thread_id`
- If not found, creates a new chat session with a new `thread_id`

### 2. **LangGraph Checkpointing**
- **Uses LangGraph's built-in MongoDB checkpointer** to automatically maintain conversation history
- When a `thread_id` is provided, LangGraph automatically loads all previous messages from its checkpoint
- The `messages` list in the agent state already contains the full conversation history
- Only the **last 10 messages** (excluding tool messages) are used for context in response generation to prevent token overflow

### 3. **Persistent Storage**
- Chat sessions are stored in **two places**:
  1. **LangGraph's checkpoint collection** (automatic) - Full conversation with all tool calls and internal state
  2. **Our custom `chat_sessions` collection** - User-friendly view with just user/assistant messages
  
- Custom `chat_sessions` document structure:
  - `user_phone`: Phone number of the user
  - `thread_id`: LangGraph conversation thread ID for continuity
  - `messages`: List of user-assistant message pairs (for reference/logging)
  - `summary`: Quick reference (first 50 chars of latest query)
  - `updated_at`: Timestamp of last update

## Files Modified

### 1. `agent_service/app/workers/chatWorker.py`
- Added `save_chat_session()` function to persist chat sessions in custom collection
- Updated `process_message()` to:
  - Check for existing chat sessions
  - Extract `thread_id` if exists
  - Pass `thread_id` to agent (LangGraph handles the rest)
  - Save/update chat session after response

### 2. `agent_service/app/agents/react_agent_v2/graph.py`
- Updated `chat_with_agent()` function to accept optional `thread_id`
- LangGraph's checkpointer automatically loads previous messages when `thread_id` is provided
- No manual history management needed

### 3. `agent_service/app/agents/react_agent_v2/state.py`
- No changes to state (removed manual `chat_history` field)
- LangGraph's `messages` field already contains full conversation history

### 4. `agent_service/app/agents/react_agent_v2/nodes.py`
- Updated `generate_response_node()` to:
  - Extract conversation history from LangGraph's `state["messages"]`
  - Filter to only human and AI messages (exclude tool messages)
  - Use last 10 messages for context to prevent token overflow
  - Include conversation history in the prompt

### 5. `agent_service/app/models/chat.py`
- Added `thread_id` field to `ChatSession` model
- Links our custom collection to LangGraph conversation threads

### 6. `agent_service/app/api/v1/chat.py`
- Updated SMS webhook handler with same logic as call handler
- Added `save_chat_session()` helper function
- Simplified to just check for `thread_id` and pass it to agent

## How It Works

### Flow Diagram
```
Incoming Call/SMS
    ↓
Check MongoDB chat_sessions for existing session by phone number
    ↓
    ├─ Session Found → Extract thread_id
    │
    └─ No Session → thread_id = None (new conversation)
    ↓
Pass (phone, query, thread_id) to chat_with_agent()
    ↓
LangGraph's MongoDB Checkpointer:
    ├─ If thread_id exists: Load all previous messages automatically
    └─ If thread_id is new: Start fresh conversation
    ↓
Agent processes with:
    - Full message history in state["messages"] (from LangGraph)
    - Last 10 user/assistant messages extracted for context
    - Knowledge base retrieval
    - Web search (if needed)
    ↓
Generate response
    ↓
Save/Update chat session in our custom collection (for easy querying)
    ↓
Send response via SMS or call
```

### LangGraph Checkpointing Architecture

```
┌─────────────────────────────────────────┐
│   LangGraph MongoDB Checkpointer        │
│  (Automatic, managed by LangGraph)      │
│                                         │
│  Collection: "checkpoints"              │
│  - Stores full agent state              │
│  - All messages (user, AI, tool)        │
│  - Internal state variables             │
│  - Keyed by thread_id                   │
└─────────────────────────────────────────┘
              ↕ (automatic)
┌─────────────────────────────────────────┐
│      Agent Execution                    │
│  state["messages"] contains full        │
│  conversation history automatically     │
└─────────────────────────────────────────┘
              ↕
┌─────────────────────────────────────────┐
│   Custom chat_sessions Collection       │
│  (Manual, for querying/logging)         │
│                                         │
│  - user_phone                           │
│  - thread_id (links to LangGraph)       │
│  - messages (user/assistant only)       │
│  - summary, updated_at                  │
└─────────────────────────────────────────┘
```

### Example Scenario

**First Call:**
- User calls: "How do I treat wheat rust?"
- System: 
  - No existing session found
  - Creates new `thread_id = "abc-123"`
  - LangGraph creates checkpoint with this thread_id
  - Saves to custom `chat_sessions` collection
- Response: Provides treatment information

**Second Call (Same Phone Number):**
- User calls: "What about prevention?"
- System:
  - Finds existing session with `thread_id = "abc-123"`
  - Passes thread_id to LangGraph
  - **LangGraph automatically loads**: "How do I treat wheat rust?" + previous response
  - Agent sees full context in `state["messages"]`
  - Extracts last 10 messages for prompt context
- Response: Provides prevention methods (with context awareness of wheat rust discussion)

## Benefits

1. **No Duplication**: LangGraph already manages conversation history - we just use it
2. **Automatic**: No manual history tracking, extraction, or passing needed
3. **Robust**: LangGraph's checkpointer is battle-tested and handles edge cases
4. **Token Efficient**: Only last 10 messages used in prompt context
5. **Persistent**: Conversations survive service restarts (MongoDB checkpointer)
6. **Dual Storage**: 
   - LangGraph checkpoint: Full detailed state for agent execution
   - Custom collection: Clean view for queries, analytics, debugging

## Configuration

Uses existing MongoDB connection settings from environment variables:
- `MONGO_URL`: MongoDB connection string
- `DB_NAME`: Database name

LangGraph automatically creates a `checkpoints` collection in the same database.

## Notes

- **LangGraph manages conversation state** - we just provide the `thread_id`
- **No manual history passing** - state["messages"] contains everything automatically
- **Last 10 messages limit** applied only during response generation for token efficiency
- **Custom chat_sessions collection** is optional but useful for:
  - Quick phone number lookups
  - User-friendly message history (without tool calls)
  - Analytics and reporting
  - Summary generation
