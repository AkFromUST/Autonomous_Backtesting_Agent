from __future__ import annotations

import json
import re
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_llm_json(content: str, model_cls: Type[T]) -> T:
    """
    Parse a Pydantic model from an LLM response that may be wrapped in markdown fences.
    Falls back to extracting the first JSON object/array found in the string.
    """
    text = content.strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # If still not clean JSON, find the first { ... } block
    if not text.startswith("{"):
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            text = brace_match.group(0)

    return model_cls.model_validate_json(text)
