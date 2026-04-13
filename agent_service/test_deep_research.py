import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from app.agents.deep_research_agent.graph import chat_with_agent


TEST_QUERIES = [
    {
        "label": "Simple greeting",
        "query": "hello",
        "history": [],
    },
    {
        "label": "Single-topic agriculture question",
        "query": "How much urea should I use for wheat in Uttar Pradesh?",
        "history": [],
    },
    {
        "label": "Multi-aspect question",
        "query": "My tomatoes are not developing the right color; what should the temperature be?",
        "history": [],
    },
    {
        "label": "Follow-up with history",
        "query": "What about Karnataka?",
        "history": [
            {"role": "user", "content": "Compare drip irrigation and flood irrigation cost for tomato farming in Maharashtra"},
            {"role": "assistant", "content": "In Maharashtra, drip irrigation for tomato costs around 30000 to 50000 rupees per acre for setup."},
        ],
    },
]


def run_single_test(test_case):
    label = test_case["label"]
    query = test_case["query"]
    history = test_case["history"]

    print(f"\n{'#'*70}")
    print(f"TEST: {label}")
    print(f"QUERY: {query}")
    if history:
        print(f"HISTORY: {len(history)} messages")
    print(f"{'#'*70}")

    start = time.time()
    try:
        result = chat_with_agent("test-user-001", query, chat_history=history)
        elapsed = time.time() - start

        response = result.get("response", "NO RESPONSE")
        print(f"\nRESPONSE ({elapsed:.1f}s):")
        print(response)
        print(f"\nStatus: PASS | Time: {elapsed:.1f}s | Length: {len(response)} chars")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\nStatus: FAIL | Time: {elapsed:.1f}s | Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) > 1:
        index = int(sys.argv[1])
        if 0 <= index < len(TEST_QUERIES):
            run_single_test(TEST_QUERIES[index])
        else:
            print(f"Invalid index. Use 0-{len(TEST_QUERIES)-1}")
        return

    print("Deep Research Agent Test Suite")
    print(f"Running {len(TEST_QUERIES)} test cases\n")

    results = []
    for i, tc in enumerate(TEST_QUERIES):
        passed = run_single_test(tc)
        results.append((tc["label"], passed))
        if i < len(TEST_QUERIES) - 1:
            time.sleep(2)

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for label, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
    total_pass = sum(1 for _, p in results if p)
    print(f"\n{total_pass}/{len(results)} passed")


if __name__ == "__main__":
    main()
