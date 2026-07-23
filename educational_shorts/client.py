from typing import TypeVar, Type

from ollama import chat
from pydantic import BaseModel, ValidationError

from educational_shorts.config import (
    MODEL,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    REPEAT_PENALTY,
    NUM_CTX,
    MAX_RETRIES,
)

T = TypeVar("T", bound=BaseModel)


def ask_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    schema: Type[T],
    temperature: float = TEMPERATURE,
    seed: int | None = None,
) -> T:
    """
    Send a prompt to the local Ollama model and return
    a validated Pydantic object.
    """

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format=schema.model_json_schema(),
                options={
                    "temperature": temperature,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "repeat_penalty": REPEAT_PENALTY,
                    "num_ctx": NUM_CTX,
                    "seed": seed,
                },
                think=False,
            )

            return schema.model_validate_json(response.message.content)

        except ValidationError as error:
            last_error = error
            print(
                f"Validation failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

        except Exception as error:
            last_error = error
            print(
                f"LLM request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {error}"
            )

    raise RuntimeError(
        f"Failed after {MAX_RETRIES} attempts.\n\n"
        f"Last error:\n{last_error}"
    )