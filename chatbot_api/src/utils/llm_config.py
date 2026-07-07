import os


def llm_provider_kwargs() -> dict:
    """Return extra kwargs for ChatOpenAI so it routes through an
    OpenAI-compatible third-party provider (e.g. NVIDIA NIM) when
    NVIDIA_API_KEY is set, otherwise fall back to OpenAI defaults.
    """
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")

    if not nvidia_api_key:
        return {}

    nvidia_base_url = os.getenv(
        "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )

    return {"api_key": nvidia_api_key, "base_url": nvidia_base_url}


def embedding_provider_kwargs(input_type: str = "passage") -> dict:
    """Return extra kwargs for OpenAIEmbeddings so it routes through
    NVIDIA NIM when NVIDIA_API_KEY is set, otherwise fall back to OpenAI
    defaults.

    Some NVIDIA embedding models (e.g. nvidia/nv-embedqa-e5-v5) require a
    non-standard `input_type` field ("query" or "passage") in the request
    body. The `openai` SDK's typed client doesn't expose this directly, so
    it's injected via `model_kwargs={"extra_body": ...}`, which
    langchain_openai forwards as-is to the underlying API call.
    """
    kwargs = llm_provider_kwargs()

    if not kwargs:
        return {}

    kwargs["model_kwargs"] = {"extra_body": {"input_type": input_type}}
    return kwargs
