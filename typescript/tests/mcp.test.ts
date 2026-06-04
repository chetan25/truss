import { describe, it, expect, vi } from 'vitest';
import { McpManifest, McpInterceptor, McpCall } from '../src/mcp/interceptor.js';
import { ToolOutOfScope } from '../src/errors.js';

describe('McpInterceptor', () => {
  it('allows allowed tool', () => {
    const i = new McpInterceptor(new McpManifest(['read_file']));
    expect(() => i.check({ toolName: 'read_file', arguments: {} })).not.toThrow();
  });

  it('blocks denied tool', () => {
    const i = new McpInterceptor(new McpManifest(['read_file']));
    expect(() => i.check({ toolName: 'write_file', arguments: {} })).toThrow(ToolOutOfScope);
  });

  it('wrap calls fn on allowed', () => {
    const i = new McpInterceptor(new McpManifest(['tool_a']));
    const result = i.wrap({ toolName: 'tool_a', arguments: { x: 1 } }, c => (c.arguments as any).x * 2);
    expect(result).toBe(2);
  });

  it('wrap does not call fn on denied', () => {
    const fn = vi.fn();
    const i = new McpInterceptor(new McpManifest(['tool_a']));
    expect(() => i.wrap({ toolName: 'tool_b', arguments: {} }, fn)).toThrow(ToolOutOfScope);
    expect(fn).not.toHaveBeenCalled();
  });

  it('empty manifest denies all', () => {
    const i = new McpInterceptor(new McpManifest([]));
    expect(() => i.check({ toolName: 'anything', arguments: {} })).toThrow(ToolOutOfScope);
  });
});
