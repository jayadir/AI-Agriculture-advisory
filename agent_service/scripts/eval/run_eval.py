import os
import sys
import json
import time

# Add the root directory to path so we can import the agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.agents.deep_research_agent.graph import chat_with_agent
from app.agents.deep_research_agent.model_registry import ModelRegistry
from pydantic import BaseModel, Field

# Pydantic schema for the Judge
class EvaluationScore(BaseModel):
    correctness: int = Field(description="Score 1-5. Does the agent answer match the ground truth context accurately?", ge=1, le=5)
    groundedness: int = Field(description="Score 1-5. Does the agent avoid hallucinating details not present in the truth?", ge=1, le=5)
    style: int = Field(description="Score 1-5. Is the agent farmer-friendly and free of complex jargon?", ge=1, le=5)
    reasoning: str = Field(description="Short sentence justifying the scores.")

JUDGE_PROMPT = """You are an objective AI Judge evaluating a Deep Research Agriculture Agent.
You will be provided with:
1. The Farmer's Question
2. The Agent's Answer
3. The absolute Ground Truth Context from the official manual.

Your job is to strictly evaluate the Agent's Answer based ONLY on the Ground Truth Context.
Score these three metrics from 1 (Worst) to 5 (Best):
1. 'correctness': Did the answer capture the core facts required by the question, as dictated by the Ground Truth?
2. 'groundedness': Did the agent invent/hallucinate specific numbers, chemical names, or dates that are NOT in the truth? (5 = strictly adhered, 1 = heavy hallucination)
3. 'style': Is it written in very simple, jargon-free Indian English suitable for a farmer?

Question:
{question}

Ground Truth Context:
{truth}

Agent's Generated Answer:
{agent_answer}
"""

def _load_json_with_comments(filepath):
    """Load a JSON file that may contain // line comments."""
    import re
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    # Strip // comments (but not inside strings)
    cleaned = re.sub(r'^\s*//.*$', '', raw, flags=re.MULTILINE)
    return json.loads(cleaned)


def run_benchmark():
    data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/eval_dataset.json"))
    out_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/eval_results.json"))
    
    try:
        dataset = _load_json_with_comments(data_file)
    except Exception as e:
        print(f"Error loading {data_file}: {e}")
        return

    print(f"Loaded {len(dataset)} evaluation queries. Starting Benchmark...")
    
    # Initialize Groq client specifically for the gpt-oss-120b judge
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    load_dotenv()
    
    judge_llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b",
        temperature=0.1,
    ).with_structured_output(EvaluationScore)
    
    # Initialize JSON and track processed IDs — appends to existing results, never overwrites
    processed_ids = set()
    existing_results = []
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
                for res in existing_results:
                    if "ID" in res:
                        processed_ids.add(res["ID"])
            except json.JSONDecodeError:
                existing_results = []
        print(f"Found {len(processed_ids)} already evaluated queries. Resuming from where we left off...")
    else:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([], f)
            
    # Process all questions — skips any already-evaluated IDs
    for i, item in enumerate(dataset):
        qid = item.get("id", f"q_{i+1}")
        
        # Skip if already evaluated
        if qid in processed_ids:
            continue
            
        print(f"\n--- Testing Query {i+1}/{len(dataset)} ---")
        question = item["question"]
        print(f"Q: {question}")
        
        # Rate limit settings for Groq Free Tier
        # ~6000 TPM or ~30 RPM limits mean we should pace our requests significantly
        
        # 1. Run the Agent (with retry for rate limits)
        max_retries = 3
        agent_success = False
        answer = "ERROR: Agent failed to run"
        latency = 0
        
        for attempt in range(max_retries):
            start_time = time.time()
            try:
                agent_response = chat_with_agent(user_id="eval_benchmark", query=question, chat_history=[])
                latency = round(time.time() - start_time, 2)
                answer = agent_response.get("response", "No response generated")
                print(f"Agent finished in {latency}s")
                agent_success = True
                break
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                    print(f"Rate limit hit during Agent run (attempt {attempt+1}/{max_retries}). Sleeping 5s...")
                    time.sleep(5)
                else:
                    print(f"Agent failed: {e}")
                    answer = f"ERROR: {str(e)}"
                    latency = 0
                    break
        
        # 2. Run the Judge (with retry for rate limits)
        score = EvaluationScore(correctness=1, groundedness=1, style=1, reasoning="Eval failed")
        if agent_success:
            for attempt in range(max_retries):
                try:
                    eval_input = JUDGE_PROMPT.format(question=question, truth=item["ground_truth_context"], agent_answer=answer)
                    score = judge_llm.invoke([{"role": "user", "content": eval_input}])
                    print(f"Score - Correct: {score.correctness}/5 | Grounded: {score.groundedness}/5 | Style: {score.style}/5")
                    print(f"Reasoning: {score.reasoning}")
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                        print(f"Rate limit hit during Judge run (attempt {attempt+1}/{max_retries}). Sleeping 5s...")
                        time.sleep(5)
                    else:
                        print(f"Judge failed: {e}")
                        break
        else:
            print("Judge skipped because agent failed.")
            
        # 3. Save progressively to JSON
        try:
            new_result = {
                "ID": qid,
                "Latency_s": latency,
                "Correctness": score.correctness,
                "Groundedness": score.groundedness,
                "Style": score.style,
                "Question": question,
                "GroundTruth": item["ground_truth_context"],
                "AgentAnswer": answer,
                "Reasoning": score.reasoning
            }
            existing_results.append(new_result)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, indent=4)
        except Exception as e:
            print(f"Failed to update JSON: {e}")
            
        # No rate limiting sleep needed for paid API
        
    print(f"\nBenchmark Complete! All {len(dataset)} results saved to {out_file}")

if __name__ == "__main__":
    run_benchmark()
