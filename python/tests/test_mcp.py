import pytest
from truss.mcp.interceptor import McpManifest, McpInterceptor, McpCall
from truss.errors import ToolOutOfScope


def test_allowed_tool_passes():
    manifest = McpManifest(allowed_tools=["read_file", "list_dir"])
    interceptor = McpInterceptor(manifest)
    interceptor.check(McpCall(tool_name="read_file", arguments={"path": "/tmp/x"}))


def test_denied_tool_raises():
    manifest = McpManifest(allowed_tools=["read_file"])
    interceptor = McpInterceptor(manifest)
    with pytest.raises(ToolOutOfScope, match="write_file"):
        interceptor.check(McpCall(tool_name="write_file", arguments={}))


def test_wrap_calls_fn_on_allowed():
    manifest = McpManifest(allowed_tools=["tool_a"])
    interceptor = McpInterceptor(manifest)
    call = McpCall(tool_name="tool_a", arguments={"x": 1})
    result = interceptor.wrap(call, lambda c: c.arguments["x"] * 2)
    assert result == 2


def test_wrap_raises_before_calling_fn_on_denied():
    called = []
    manifest = McpManifest(allowed_tools=["tool_a"])
    interceptor = McpInterceptor(manifest)
    with pytest.raises(ToolOutOfScope):
        interceptor.wrap(McpCall(tool_name="tool_b", arguments={}), lambda c: called.append(True))
    assert not called


def test_empty_manifest_denies_all():
    manifest = McpManifest(allowed_tools=[])
    interceptor = McpInterceptor(manifest)
    with pytest.raises(ToolOutOfScope):
        interceptor.check(McpCall(tool_name="any_tool", arguments={}))


def test_manifest_is_allowed():
    manifest = McpManifest(allowed_tools=["a", "b"])
    assert manifest.is_allowed("a") is True
    assert manifest.is_allowed("c") is False
