# Local Agent Testing Guide

Test your complete agent flow without AWS SQS or Twilio dependencies.

## Testing Scripts

### 1. Quick Test (No Database)
Fast single-query test without database persistence.

```powershell
# Default test query
python quick_test.py

# Custom query
python quick_test.py "What are the symptoms of wheat rust?"
```

**Use when:**
- Quick functionality check
- Testing RAG retrieval
- No need for chat history

---

### 2. Local Agent Tester (Full Flow)
Complete agent testing with database persistence and chat history.

#### Interactive Mode
```powershell
python test_agent_local.py
```

**Features:**
- Chat with the agent interactively
- Maintains conversation history
- Saves to MongoDB
- Type `quit` to exit
- Type `history` to view chat
- Type `new` to start fresh session

#### Single Query Mode
```powershell
python test_agent_local.py --query "How do I treat wheat rust?"
```

#### Test Mode (Multiple Queries)
```powershell
python test_agent_local.py --test
```

Runs predefined test queries:
1. "What are the symptoms of wheat rust?"
2. "How do I treat it?"
3. "What about fertilizer recommendations?"

#### View Chat History
```powershell
python test_agent_local.py --history
```

#### Clear Chat Session
```powershell
python test_agent_local.py --clear
```

---

## Test Flow

Both scripts simulate the complete production flow:

```
User Query
    ↓
Agent Processing
    ├── RAG Engine (retrieves 5 variants × 10 docs)
    ├── Query Expansion (para, broad, tech, expl)
    ├── Tool Selection
    └── Response Generation
    ↓
Save to Database (test_agent_local.py only)
    ↓
Background Learning (optional)
    ↓
Response Returned
```

---

## Example Session

### Quick Test
```powershell
PS> python quick_test.py "What causes leaf spot in tomatoes?"

======================================================================
QUICK AGENT TEST
======================================================================

Query: What causes leaf spot in tomatoes?

Processing...

Response time: 3.42s

======================================================================
RESPONSE:
======================================================================
Leaf spot in tomatoes is typically caused by fungal pathogens...
======================================================================
```

### Interactive Mode
```powershell
PS> python test_agent_local.py

======================================================================
LOCAL AGENT TESTER
======================================================================

Initializing database connection...
Database connected

Test user exists: +1234567890

======================================================================
INTERACTIVE MODE
======================================================================
Type your queries (or 'quit' to exit, 'history' to view chat)
======================================================================

You: What is wheat rust?

======================================================================
USER QUERY: What is wheat rust?
======================================================================

Starting new chat session

Processing with agent...

Agent response time: 3.21s

----------------------------------------------------------------------
AGENT RESPONSE:
----------------------------------------------------------------------
Wheat rust is a fungal disease that affects wheat crops...
----------------------------------------------------------------------

Chat session created

Triggering background learning...

You: How do I prevent it?

======================================================================
USER QUERY: How do I prevent it?
======================================================================

Continuing existing session (Thread ID: abc123...)
Chat history: 2 messages

Processing with agent...
[Agent uses previous context about wheat rust]
...
```

---

## Configuration

### Required
- MongoDB running (for test_agent_local.py)
- FAISS index built
- Query expander weights loaded
- Embedding model configured (EMBEDDING_MODEL env var)

### Optional
- Set FORCE_CPU=1 for CPU-only mode
- Adjust RETRIEVAL_K in engine.py for more/fewer docs

---

## Differences from Production

| Aspect | Production | Local Testing |
|--------|-----------|---------------|
| Input | AWS SQS Queue | Direct function call |
| Output | Twilio SMS/Call | Console print |
| User ID | Real phone number | Fixed test number |
| Database | Async MongoDB | Async MongoDB |
| Agent Logic | Identical | Identical |
| RAG Engine | Identical | Identical |
| Learning | Background task | Background task |

---

## Troubleshooting

### MongoDB Connection Error
```
Ensure MongoDB is running:
- Check connection string in .env or config
- Default: mongodb://localhost:27017
```

### Agent Import Error
```
Verify agent code exists:
- app/agents/react_agent_v2/graph.py
- app/agents/react_agent_v2/learning.py
```

### RAG Engine Error
```
Check FAISS index and model files:
- artifacts/vector_db/agri_faiss_index/
- artifacts/models/query_expander.pth (BGE)
- artifacts/models/deep_residual_expander.pt (Jina)
```

### Slow Response
```
First query loads models (10-30s)
Subsequent queries are faster (2-5s)
Use FORCE_CPU=1 if GPU has issues
```

---

## Testing Checklist

- [ ] Quick test with default query works
- [ ] Quick test with custom query works
- [ ] Interactive mode starts successfully
- [ ] Agent returns responses
- [ ] Chat history persists across queries
- [ ] Context is maintained in conversation
- [ ] Background learning triggers (check logs)
- [ ] Database stores chat sessions
- [ ] Can view history with --history
- [ ] Can clear session with --clear

---

## Next Steps

After local testing succeeds:

1. **Deploy to Production**: Connect AWS SQS and Twilio
2. **Add Re-ranker**: Implement document re-ranking in engine.py
3. **Monitor Performance**: Track response times and accuracy
4. **Tune Parameters**: Adjust RETRIEVAL_K, MAX_CONTEXT_DOCS

---

## Support

Common test queries to try:
- "What are the symptoms of wheat rust?"
- "How do I control aphids in my cotton crop?"
- "What fertilizer should I use for rice?"
- "My tomato leaves are turning yellow, what should I do?"
- "Tell me about crop rotation benefits"
