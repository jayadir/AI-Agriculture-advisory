import sys
import os
from operator import itemgetter
from typing import List, Optional,Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exceptions.c_exception import DocumentPortalException
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from pydantic import ValidationError

class ConversationalRAG:
    
    def __init__(self,session_id:Optional[str],retriever=None):
        try:
            self.session_id=session_id
            self.llm=self._load_llm()
            self.contextualize_prompt:ChatPromptTemplate=PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt:ChatPromptTemplate=PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            self.retriever=retriever
            self.chain=None
            if self.retriever is not None:
                self._build_lcel_chain()
                
            log.info("ConversationalRAG initialized",session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys) 
    
    def load_retriever_from_faiss(
        self,
        index_path:str,
        k:int=5,
        index_name:str="index",
        search_type:str="mmr",
        fetch_k:int=20,
        lambda_mult:float=0.5,
        search_kwargs:Optional[Dict[str,Any]]=None
    ):
        try:
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index file not found at {index_path}")
            embeddings=ModelLoader().load_embeddings()
            vetorstore=FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True
            )
            
            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs["fetch_k"] = fetch_k
                    search_kwargs["lambda_mult"] = lambda_mult
            self.retriever=vetorstore.as_retriever(search_type=search_type,search_kwargs=search_kwargs)
            self._build_lcel_chain()
            log.info(
                "FAISS retriever loaded successfully",
                index_path=index_path,
                index_name=index_name,
                search_type=search_type,
                k=k,
                fetch_k=fetch_k if search_type == "mmr" else None,
                lambda_mult=lambda_mult if search_type == "mmr" else None,
                session_id=self.session_id,
            )
            return self.retriever
        except Exception as e:
            log.error("Failed to load FAISS retriever", error=str(e))
            raise DocumentPortalException("Error loading FAISS retriever", sys)
    
    def invoke(self,user_input:str,chat_history:Optional[List[BaseMessage]]=None) ->str:
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload={"input":user_input,"chat_history":chat_history}
            answer=self.chain.invoke(payload)
            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."
            
            try:
                validated=ChatAnswer(answer=str(answer))
                answer= validated.answer
            except ValidationError as ve:
                log.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid chat answer", sys)
            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)
        
    def _load_llm(self):
        try:
            llm=ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM loading failed")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)   
    
    @staticmethod
    def _format_docs(docs)->str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)
    
    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("Retriever is not set", sys)
            
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )
            
            retrieve_docs = question_rewriter | self.retriever | self._format_docs
            
            self.chain=(
                 {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )
            
            log.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)

# Backward-compatibility alias (handle legacy import/typo)
ConverstionalRAG = ConversationalRAG