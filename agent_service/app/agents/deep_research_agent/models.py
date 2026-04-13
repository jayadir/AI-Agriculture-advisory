from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PlanDecision(BaseModel):
    route: Literal["direct", "research"] = Field(
        description="Whether to respond directly or do research"
    )
    direct_response: Optional[str] = Field(
        default=None,
        description="Response text when route is direct"
    )
    sub_questions: Optional[List[str]] = Field(
        default=None,
        description="1-3 focused sub-questions for research"
    )
    search_strategies: Optional[List[Literal["web_only", "kb_then_web"]]] = Field(
        default=None,
        description="Strategy per sub-question"
    )
    india_localization: Optional[List[str]] = Field(
        default=None,
        description="Region or crop context per sub-question"
    )
    worker_instructions: Optional[List[str]] = Field(
        default=None,
        description="Dynamic reasoning-based instructions or focus areas for the worker per sub-question"
    )
    search_keywords: Optional[List[str]] = Field(
        default=None,
        description="Optimized academic/extension search terms per sub-question for better web results"
    )


class WorkerReport(BaseModel):
    sub_question: str
    answer_found: bool
    findings: str
    confidence: Literal["high", "medium", "low"]
    sources: List[str] = Field(default_factory=list)
    data_points: Optional[List[str]] = None
    contradictions: Optional[str] = None


class SynthesisResult(BaseModel):
    synthesized_answer: str
    overall_confidence: Literal["high", "medium", "low"]
    verification_notes: str
    sources_used: List[str] = Field(default_factory=list)
    needs_retry: bool = False
    retry_questions: Optional[List[str]] = None
