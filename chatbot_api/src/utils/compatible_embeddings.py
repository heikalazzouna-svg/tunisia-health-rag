from typing import List, Optional

from langchain_openai import OpenAIEmbeddings


class SimpleOpenAICompatibleEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings subclass that always sends raw text to the
    embeddings endpoint instead of tiktoken-encoded integer token arrays.

    langchain_openai.OpenAIEmbeddings pre-tokenizes text with tiktoken and
    submits `input` as arrays of integer token IDs by default -- a
    real-OpenAI-only optimization that many OpenAI-compatible providers
    (e.g. NVIDIA NIM) don't support, causing server-side errors like
    "'list' object has no attribute 'strip'". This subclass bypasses that
    logic entirely and sends plain text, which every OpenAI-compatible
    provider (including OpenAI itself) accepts.
    """

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        response = self.client.create(input=texts, **self._invocation_params)
        if not isinstance(response, dict):
            response = response.model_dump()
        return [item["embedding"] for item in response["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
