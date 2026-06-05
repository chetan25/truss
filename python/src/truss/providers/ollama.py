from truss.providers.base import LLMMessage, LLMResponse


class OllamaProvider:
    def complete(self, messages: list[LLMMessage], model: str = "llama3", **opts) -> LLMResponse:
        raise NotImplementedError(
            "OllamaProvider is not yet implemented. "
            "Track progress at github.com/your-org/truss — planned for Phase 3."
        )
