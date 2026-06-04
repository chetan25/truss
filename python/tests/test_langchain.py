import pytest
from truss.adapters.langchain import TrussMemory


def test_save_context_adds_two_blocks():
    mem = TrussMemory(target_tokens=1000)
    mem.save_context({"input": "What is the weather?"}, {"output": "It is sunny."})
    assert len(mem.blocks) == 2


def test_load_memory_variables_returns_history_key():
    mem = TrussMemory(target_tokens=1000)
    mem.save_context({"input": "Hello"}, {"output": "World"})
    vars_ = mem.load_memory_variables({})
    assert "history" in vars_
    assert "Human:" in vars_["history"]
    assert "AI:" in vars_["history"]


def test_compression_fires_when_over_budget():
    mem = TrussMemory(target_tokens=20, preserve_recent=1)
    for i in range(30):
        mem.save_context({"input": f"q{i} " + "x" * 100}, {"output": f"a{i} " + "y" * 100})
    vars_ = mem.load_memory_variables({})
    assert len(vars_["history"]) < 30 * 200


def test_clear_empties_blocks():
    mem = TrussMemory()
    mem.save_context({"input": "hi"}, {"output": "hello"})
    mem.clear()
    assert len(mem.blocks) == 0


def test_memory_key_attribute():
    mem = TrussMemory()
    assert mem.memory_key == "history"


def test_custom_memory_key():
    mem = TrussMemory(memory_key="chat_history")
    mem.save_context({"input": "hi"}, {"output": "hello"})
    vars_ = mem.load_memory_variables({})
    assert "chat_history" in vars_
