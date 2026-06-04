/**
 * Hermes Agent — Truss TypeScript Phase 1 reference example.
 * Run: npx tsx examples/hermes.ts  (from typescript/)
 */
import {
  Session, makeContextBlock, makeEnvelope,
  ContextRole, ContextWeight, ModelTier,
  McpManifest, McpInterceptor,
  ModelSpec, RouterConfig, RouterRule, route,
  pack, BudgetCarve,
} from './src/index.js';

const parent = makeEnvelope('Research cheapest S3-compatible cloud storage for 10TB');
parent.budgetUsdRemaining = 1.0;
parent.context = [
  makeContextBlock(ContextRole.Task, ContextWeight.Critical, 'Find cheapest S3-compatible storage for 10TB dataset under $500/month.', 'user'),
  makeContextBlock(ContextRole.Constraint, ContextWeight.Critical, 'Must be S3-compatible. Budget: $500/month max.', 'user'),
  makeContextBlock(ContextRole.Finding, ContextWeight.High, 'Backblaze B2: $6/TB/month, S3-compatible.', 'search'),
  makeContextBlock(ContextRole.Finding, ContextWeight.High, 'Cloudflare R2: $15/TB/month, zero egress fees.', 'search'),
  makeContextBlock(ContextRole.Finding, ContextWeight.Normal, 'AWS S3 Standard: $23/TB/month, widest ecosystem.', 'search'),
  makeContextBlock(ContextRole.Background, ContextWeight.Background, 'Cloud storage history dates to the 1960s mainframe era.', 'wikipedia'),
];

const models: ModelSpec[] = [
  { name: 'claude-haiku-4-5', tier: ModelTier.Cheap, maxTokens: 8192, costPer1kInput: 0.001, costPer1kOutput: 0.005 },
  { name: 'claude-sonnet-4-6', tier: ModelTier.Standard, maxTokens: 16384, costPer1kInput: 0.003, costPer1kOutput: 0.015 },
];
const selected = route('Find cheapest storage option', { models, rules: [{ keywords: ['cheapest'], preferredTier: ModelTier.Cheap }], defaultTier: ModelTier.Standard });

const interceptor = new McpInterceptor(new McpManifest(['search_web', 'read_url']));
const child = pack(parent, 'Rank options by price', [ContextWeight.Critical, ContextWeight.High], BudgetCarve.percent(0.3));

const session = new Session({ envelope: parent, budgetUsd: 1.0, targetTokens: 150, preserveRecent: 2 });
const result = session.compress(parent.context);
session.checkpoint('after compression');
session.recordUsage(result.tokensAfter, 80, 0.005, selected.name);

try {
  interceptor.check({ toolName: 'search_web', arguments: { q: 'cheapest cloud storage' } });
  console.log('MCP: search_web allowed ✓');
} catch (e) { console.log(`MCP blocked: ${e}`); }

try {
  interceptor.check({ toolName: 'delete_file', arguments: {} });
} catch (e) { console.log(`MCP blocked delete_file ✓ (${e})`); }

const report = session.report();
console.log(`\n=== Compression ===\nBlocks: ${parent.context.length} -> ${result.blocks.length} (saved ${result.tokensSaved} tokens)`);
console.log(`\n=== Child Envelope ===\nTask: ${child.task}\nContext blocks: ${child.context.length}\nBudget carved: ${child.budgetUsdRemaining?.toFixed(2) ?? 'unlimited'}`);
console.log(`\n=== Session Report ===\nSession: ${report.sessionId.slice(0, 8)}\nContext: ${report.tokensBefore} -> ${report.tokensAfter} (saved ${report.tokensSaved})\nBudget: $${report.budgetUsedUsd.toFixed(4)} of $${report.budgetLimitUsd?.toFixed(2)} used\nCheckpoints: ${report.checkpointCount}`);
