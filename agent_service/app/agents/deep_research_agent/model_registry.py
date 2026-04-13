import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class ModelRegistry:
    _models = {}

    @classmethod
    def get(cls, role):
        if role not in cls._models:
            cls._models[role] = cls._create(role)
        return cls._models[role]

    @classmethod
    def _create(cls, role):
        if role == "responder":
            from app.services.llm import _get_model
            return _get_model()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY")

        configs = {
            "planner": {
                "model": "openai/gpt-oss-120b",
                "temperature": 0.1,
            },
            "worker": {
                "model": "openai/gpt-oss-20b",
                "temperature": 0,
            },
            "synthesizer": {
                "model": "openai/gpt-oss-120b",
                "temperature": 0.1,
            },
            "grader": {
                "model": "openai/gpt-oss-20b",
                "temperature": 0,
            },
        }

        cfg = configs[role]
        env_key = f"GROQ_MODEL_{role.upper()}"
        model_name = os.getenv(env_key, cfg["model"])

        return ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=cfg["temperature"],
        )
