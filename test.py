import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from langchain_core.messages import HumanMessage,AIMessage
load_dotenv()

def test_rag_system():
    
    try:
        test_files=[
            "C:/Users/JAYADIR/OneDrive/Desktop/llmops/data/FUNDAMENTALS-OF-CROP-PHYSIOLOGY.pdf"
        ]
        uploaded_files=[]
        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path,"rb"))
            else:
                print(f"File not found: {file_path}")
        if not uploaded_files:
            print("No valid files to ingest. Exiting test.")
            sys.exit(1)
        ci=ChatIngestor(
            temp_base="data",
            faiss_base="faiss_index",
            use_session_dirs=True
        )
        
        retriever=ci.built_retriver(
            uploaded_files, 
            chunk_size=700, 
            chunk_overlap=100, 
            k=5,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )
        
        for f in uploaded_files:
            try:
                f.close()
            except Exception as e:
                pass
        session_id=ci.session_id
        index_dir = os.path.join("faiss_index", session_id)
        rag=ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir, 
            k=5, 
            index_name=os.getenv("FAISS_INDEX_NAME", "index"),
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )
        chat_history = []
        print("\nType 'exit' to quit the chat.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                print("Goodbye!")
                break

            answer = rag.invoke(user_input, chat_history=chat_history)
            print("Assistant:", answer)

            # Maintain conversation history for context in subsequent turns
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_rag_system()
        