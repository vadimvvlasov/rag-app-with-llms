"""
LLMClient implementations for the RAG pipeline.

Three backends are provided:
- OpenAIClient:      calls the OpenAI API (key from OPENAI_API_KEY env var)
- OllamaClient:      calls a local Ollama REST API
- OpenRouterClient:  calls the OpenRouter API (key from OPENROUTER_API_KEY env var)

All three implement the LLMClient Protocol defined in interfaces.py.
"""

import os

import requests
from openai import OpenAI


class OpenAIClient:
    """LLMClient that calls the OpenAI Responses API.

    Uses ``client.responses.create`` — the current preferred API over the
    legacy Chat Completions endpoint.

    The API key is read exclusively from the ``OPENAI_API_KEY`` environment
    variable (Req 5.1).  Any API error is propagated to the caller without
    wrapping (Req 3.4).
    """

    def __init__(self) -> None:
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def complete(self, prompt: str, instructions: str, model: str) -> str:
        """Send a prompt to OpenAI via the Responses API and return the text.

        Passes instructions as a ``developer`` role message and the prompt as
        a ``user`` role message inside the ``input`` list.

        Args:
            prompt:       User-facing message (question + context).
            instructions: System-level instructions sent as the ``developer``
                          role message.
            model:        OpenAI model identifier, e.g. ``"gpt-4o-mini"``.

        Returns:
            The model's text response as a plain string.

        Raises:
            openai.APIError: If the OpenAI API returns an error (Req 3.4).
        """
        response = self._client.responses.create(
            model=model,
            input=[
                {"role": "developer", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )
        return response.output_text


class OllamaClient:
    """LLMClient that calls a locally-running Ollama instance via its REST API.

    Args:
        base_url: Base URL of the Ollama server.
                  Defaults to ``"http://localhost:11434"``.
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")

    def complete(self, prompt: str, instructions: str, model: str) -> str:
        """Send a prompt to Ollama and return the assistant's text response.

        Args:
            prompt:       User-facing message (question + context).
            instructions: System message that guides model behaviour.
            model:        Ollama model identifier, e.g. ``"llama3"``.

        Returns:
            The model's text response as a plain string.

        Raises:
            requests.HTTPError: If the Ollama API returns a non-2xx status.
        """
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


class OpenRouterClient:
    """LLMClient that calls the OpenRouter API via the Responses API.

    Uses ``client.responses.create`` pointed at ``https://openrouter.ai/api/v1``.
    The API key is read from the ``OPENROUTER_API_KEY`` environment variable.
    """

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    def complete(self, prompt: str, instructions: str, model: str) -> str:
        """Send a prompt to OpenRouter via the Responses API and return the text.

        Passes instructions as a ``developer`` role message and the prompt as
        a ``user`` role message inside the ``input`` list.

        Args:
            prompt:       User-facing message (question + context).
            instructions: System-level instructions sent as the ``developer``
                          role message.
            model:        Model identifier supported by OpenRouter,
                          e.g. ``"openai/gpt-4o-mini"``.

        Returns:
            The model's text response as a plain string.

        Raises:
            openai.APIError: If the OpenRouter API returns an error.
        """
        response = self._client.responses.create(
            model=model,
            input=[
                {"role": "developer", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )
        return response.output_text
