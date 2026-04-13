from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from app.agents.deep_research_agent.model_registry import ModelRegistry
from app.agents.deep_research_agent.state import DeepResearchState


class ResponseOutput(BaseModel):
    content: str = Field(description="The farmer-friendly response text")


RESPONDER_SYSTEM_PROMPT = """You are an agriculture expert assistant specialized in Indian farming conditions.

RULES:
1. Use ONLY the verified research data provided below. Do not make up facts.
2. Ensure the response is grounded in Indian farming context.
3. Your audience is a village farmer with no formal education. Do not use ANY technical
   terms, scientific jargon, or complex agricultural concepts.
4. Plain text only. Write the ENTIRE response as a single, continuous, conversational paragraph. Do NOT use pointwise answers, bullet points, numbered lists, or new lines.
5. The response will be played back as audio. Keep it natural and conversational,
   like a village agriculture officer speaking slowly and clearly.
6. Avoid symbols, formulas, and chemistry abbreviations such as N, P, K, P2O5, K2O.
7. Do NOT convert units or quantities. If the research data contains units like kg/ha,
   g/l, ppm, or hectare/acre, repeat them exactly as written.
8. Prefer simple spoken names like urea, DAP, and potash when they appear in the research.
10. Keep sentences short. Speak as if explaining to someone who has never gone to school.
11. If a technical term cannot be simplified, explain it in one short sentence first.
12. Prefer simple conversational Indian English suitable for audio playback.
13. If the confidence level is low, honestly tell the farmer that you are not fully sure
    and suggest they consult their local agriculture office.

CRITICAL ANTI-HALLUCINATION RULES:
14. NEVER invent or guess dosages, measurements, or timeframes. If the farmer explicitly asks for a specific number (like "how much fertilizer") and the research data does not contain it, state honestly that you do not have the exact amount, but still provide any other helpful, verified qualitative advice from the research. If the farmer asks a "how to" or qualitative question, just give the recommended steps without apologizing for missing numbers.
15. If a piece of data is marked as [UNVERIFIED], [LOW_CONFIDENCE_SOURCE], [NO_SOURCE],
    or DATA_NOT_FOUND in the research, do NOT present that claim as a fact.
    Either skip it or say clearly you do not have confirmed information.
16. Remove all [Source: ...] citations and internal evidence labels from your response,
    but do NOT keep the underlying claim if it was marked [NO_SOURCE] or [LOW_CONFIDENCE_SOURCE].
17. Do not add generic farming advice (like "keep soil moist" or "use good seeds")
    unless it is specifically mentioned in the research data provided."""


def responder_node(state: DeepResearchState) -> dict:
    print("Running responder")

    llm = ModelRegistry.get("responder")
    synthesis = state.get("synthesis", {})
    user_query = state["user_query"]
    chat_history = state.get("chat_history", [])

    synthesized_answer = synthesis.get("synthesized_answer", "")
    confidence = synthesis.get("overall_confidence", "low")
    verification_notes = synthesis.get("verification_notes", "")

    history_context = ""
    if chat_history:
        lines = []
        for msg in chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        if lines:
            history_context = "Previous Conversation:\n" + "\n".join(lines) + "\n\n"

    user_prompt = (
        f"{history_context}"
        f"Verified Research Data:\n{synthesized_answer}\n\n"
        f"Confidence Level: {confidence}\n"
        f"Verification Notes: {verification_notes}\n\n"
        f"Farmer's Question: {user_query}"
    )

    messages = [
        {"role": "system", "content": RESPONDER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        structured_llm = llm.with_structured_output(ResponseOutput)
        response = structured_llm.invoke(messages)
        final_response = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"Structured response failed: {e}, using fallback")
        try:
            response = llm.invoke(messages)
            final_response = response.content.strip()
        except Exception as fallback_e:
            print(f"Response generation error: {fallback_e}")
            final_response = "I am sorry, there was a problem generating the answer. Please try again."

    ai_msg = AIMessage(content=final_response.strip())
    state_messages = state.get("messages", [])
    state_messages.append(ai_msg)

    print("Responder: done")

    return {
        "messages": state_messages,
        "final_response": final_response.strip(),
    }
