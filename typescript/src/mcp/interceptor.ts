import { ToolOutOfScope } from '../errors.js';

export interface McpCall {
  toolName: string;
  arguments: Record<string, unknown>;
}

export class McpManifest {
  constructor(public readonly allowedTools: string[]) {}
  isAllowed(toolName: string): boolean { return this.allowedTools.includes(toolName); }
}

export class McpInterceptor {
  constructor(private manifest: McpManifest) {}

  check(call: McpCall): void {
    if (!this.manifest.isAllowed(call.toolName)) {
      throw new ToolOutOfScope(`${call.toolName} denied by manifest`);
    }
  }

  wrap<T>(call: McpCall, fn: (call: McpCall) => T): T {
    this.check(call);
    return fn(call);
  }
}
