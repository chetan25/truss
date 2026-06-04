"""
Hermes Agent — Truss Python Phase 1 reference example.
Run from python/ directory: python -m examples.hermes.main
"""
import asyncio
from truss import (
    Session, ContextBlock, ContextRole, ContextWeight,
    AgentEnvelope, ModelTier,
    McpManifest, McpInterceptor, McpCall,
    ModelSpec, RouterConfig, RouterRule, route,
    pack, BudgetCarve,
)


async def main() -> None:
    parent = AgentEnvelope(task="Research cheapest S3-compatible cloud storage for 10TB", budget_usd_remaining=1.0)
    parent.context = [
        ContextBlock(role=ContextRole.TASK, weight=ContextWeight.CRITICAL,
                     content="Find cheapest S3-compatible storage for 10TB dataset under $500/month.", source="user"),
        ContextBlock(role=ContextRole.CONSTRAINT, weight=ContextWeight.CRITICAL,
                     content="Must be S3-compatible. Budget: $500/month max.", source="user"),
        ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.HIGH,
                     content="Backblaze B2: $6/TB/month, S3-compatible.", source="search"),
        ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.HIGH,
                     content="Cloudflare R2: $15/TB/month, zero egress fees.", source="search"),
        ContextBlock(role=ContextRole.FINDING, weight=ContextWeight.NORMAL,
                     content="AWS S3 Standard: $23/TB/month, widest ecosystem.", source="search"),
        ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND,
                     content="Cloud storage history dates to the 1960s mainframe era.", source="wikipedia"),
        ContextBlock(role=ContextRole.BACKGROUND, weight=ContextWeight.BACKGROUND,
                     content="IBM 305 RAMAC was the first commercial hard disk drive, 1956.", source="wikipedia"),
    ]

    models = [
        ModelSpec("claude-haiku-4-5", ModelTier.CHEAP, 8192, 0.001, 0.005),
        ModelSpec("claude-sonnet-4-6", ModelTier.STANDARD, 16384, 0.003, 0.015),
    ]
    router_config = RouterConfig(
        models=models,
        rules=[RouterRule(keywords=["cheapest", "summarise"], preferred_tier=ModelTier.CHEAP)],
    )
    selected_model = route("Find cheapest storage option", router_config)

    manifest = McpManifest(allowed_tools=["search_web", "read_url"])
    interceptor = McpInterceptor(manifest)

    child = pack(parent, "Rank storage options by price",
                 carry_weights=[ContextWeight.CRITICAL, ContextWeight.HIGH],
                 budget_carve=BudgetCarve.percent(0.3))

    async with Session(envelope=parent, budget_usd=1.0, target_tokens=150, preserve_recent=2) as s:
        result = s.compress(parent.context)
        cp_id = s.checkpoint("after compression")
        s.record_usage(input_tokens=result.tokens_after, output_tokens=80, cost_usd=0.005, model=selected_model.name)

        try:
            interceptor.check(McpCall(tool_name="search_web", arguments={"q": "cheapest cloud storage"}))
            print("MCP: search_web allowed")
        except Exception as e:
            print(f"MCP blocked: {e}")

        try:
            interceptor.check(McpCall(tool_name="delete_file", arguments={"path": "/important.db"}))
        except Exception as e:
            print(f"MCP blocked delete_file ({e})")

        print(f"\n=== Compression ===")
        print(f"Blocks: {len(parent.context)} -> {len(result.blocks)} (saved {result.tokens_saved} tokens)")

        print(f"\n=== Child Envelope ===")
        print(f"Task: {child.task}")
        print(f"Context blocks carried: {len(child.context)}")
        budget_str = f"${child.budget_usd_remaining:.2f}" if child.budget_usd_remaining is not None else "unlimited"
        print(f"Budget carved: {budget_str}")

        print(f"\n=== Session Report ===")
        print(s.report())


if __name__ == "__main__":
    asyncio.run(main())
