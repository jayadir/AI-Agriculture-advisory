from typing import Any
from bson import ObjectId
from pydantic_core import core_schema

class PyObjectId(str):
    """
    Custom Pydantic type to handle MongoDB ObjectId.
    It accepts both str and ObjectId, but serializes to str for JSON responses.
    """
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: Any
    ) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                # Accept an actual ObjectId instance
                core_schema.is_instance_schema(ObjectId),
                # Or accept a string and convert it to ObjectId
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(ObjectId),
                ]),
            ]),
            # Serialize back to string for the API response
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )