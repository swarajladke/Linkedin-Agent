"""Structured LLM client protocol and OpenAI implementation with bounded validation retries."""

import json
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from pilot.config import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMExtractionError(Exception):
    """Raised when an LLM fails to return valid structured data conforming to schema."""


class StructuredLLMClient(Protocol):
    """Protocol for provider-agnostic structured LLM completion."""

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        """Extract structured data adhering strictly to the given Pydantic schema."""
        ...


class OpenAIStructuredClient:
    """Synchronous OpenAI structured outputs client with lazy SDK loading and retry repair."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 2,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or (
            settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        )
        self.model = model or settings.openai_model
        self.max_retries = max_retries

        # Lazy import of OpenAI SDK
        import openai

        if not self.api_key:
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=self.api_key)

    def complete_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        """Call OpenAI chat completions with structured output, retrying on validation failures."""
        if not self.client:
            raise LLMExtractionError(
                "OpenAI API key is missing or not configured. Cannot call OpenAIStructuredClient."
            )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        json_schema = schema.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": json_schema,
            },
        }

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0.0,
                )

                content = response.choices[0].message.content or ""
                parsed = json.loads(content)
                return schema.model_validate(parsed)

            except (json.JSONDecodeError, ValidationError) as err:
                last_error = err
                if attempt < self.max_retries:
                    # Append failed assistant content and error correction prompt
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content if "content" in locals() else "",
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous response was invalid. Error details: {err}.\n"
                                f"Please fix the error and output valid JSON conforming strictly to schema."
                            ),
                        }
                    )
                else:
                    break
            except Exception as exc:
                raise LLMExtractionError(f"OpenAI API request failed: {exc}") from exc

        raise LLMExtractionError(
            f"Failed to produce valid structured output for {schema.__name__} after "
            f"{self.max_retries} retries: {last_error}"
        ) from last_error
