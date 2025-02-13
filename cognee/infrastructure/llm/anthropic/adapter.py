from typing import Type
from pydantic import BaseModel
import instructor
import anthropic
import os

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.config import get_llm_config


class AnthropicAdapter(LLMInterface):
    """Adapter for Anthropic API"""

    name = "Anthropic"
    model: str
    api_key: str

    def __init__(self, max_tokens: int, model: str = None):
        llm_config = get_llm_config()
        self.api_key = llm_config.llm_api_key
        # log api_key
        print("Anthropic API key: ", self.api_key)
        
        if not self.api_key:
            raise InvalidValueError(message="Anthropic API key is not set")

        self.aclient = instructor.from_anthropic(
            client=anthropic.AsyncAnthropic(api_key=self.api_key),
            mode=instructor.Mode.ANTHROPIC_TOOLS
        )
        self.model = model
        self.max_tokens = max_tokens

    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query."""

        return await self.aclient.messages.create(
            model=self.model,
            max_tokens=4096,
            max_retries=5,
            messages=[
                {
                    "role": "user",
                    "content": f"""Use the given format to extract information
                from the following input: {text_input}. {system_prompt}""",
                }
            ],
            response_model=response_model,
        )

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""

        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise InvalidValueError(message="No system prompt path provided.")

        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )

        return formatted_prompt
