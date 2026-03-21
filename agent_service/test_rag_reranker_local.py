"""
Local RAG Reranker Test
=======================
Runs a single query through `RAGEngine.process()` and prints the
selected reranked documents for quick evaluation.

Usage:
	python test_rag_reranker_local.py
	python test_rag_reranker_local.py "What fertilizer is best for wheat?"
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.rag.engine import get_rag_engine


def print_section(title: str):
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80)


async def run_rag_test(query: str):
	print_section("LOCAL RAG RERANKER TEST")
	print(f"Query: {query}")

	start_time = datetime.now()

	try:
		engine = await get_rag_engine()
		result = await engine.process(query)
		elapsed = (datetime.now() - start_time).total_seconds()

		print(f"\nCompleted in {elapsed:.2f}s")

		if "response" in result and result.get("source") == "error":
			print_section("ENGINE ERROR")
			print(result["response"])
			return result

		print_section("TOP RERANKED DOCS")
		reranked_docs = result.get("reranked_top_docs", [])
		if not reranked_docs:
			print("No reranked docs returned.")
		else:
			for item in reranked_docs:
				print(f"Rank: {item['rank']}")
				print(f"Score: {item['score']:.4f}")
				print(f"Source: {item['source']}")
				print(f"Preview: {item['preview']}")
				print("-" * 80)

		print_section("FINAL CONTEXT SENT TO AGENT")
		print(result.get("response_docs", "No context returned."))

		print_section("SUMMARY")
		print(f"Candidate count: {result.get('num_candidates', 0)}")
		print(f"Selected docs: {len(reranked_docs)}")

		return result

	except Exception as error:
		print_section("TEST FAILED")
		print(f"Error: {error}")
		import traceback
		traceback.print_exc()
		return None


if __name__ == "__main__":
	if len(sys.argv) > 1:
		test_query = " ".join(sys.argv[1:])
	else:
		test_query = "What are the common diseases in wheat crops and how can farmers manage them?"

	asyncio.run(run_rag_test(test_query))
