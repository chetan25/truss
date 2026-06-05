from truss.providers.base import LLMMessage, LLMResponse


class GoogleProvider:
    def complete(self, messages: list[LLMMessage], model: str = "gemini-pro", **opts) -> LLMResponse:
        raise NotImplementedError(
            "GoogleProvider is not yet implemented. "
            "Track progress at github.com/your-org/truss — planned for Phase 3."
        )
