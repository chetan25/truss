import { v4 as uuidv4 } from 'uuid';

export type UUID = string;

export function estimateTokens(text: string): number {
  return text.length === 0 ? 0 : Math.ceil(text.length / 4);
}

export enum ContextRole {
  Task = 'task',
  Constraint = 'constraint',
  Finding = 'finding',
  Decision = 'decision',
  Warning = 'warning',
  Background = 'background',
}

export enum ContextWeight {
  Background = 0,
  Normal = 1,
  High = 2,
  Critical = 3,
}

export interface ContextBlock {
  id: UUID;
  role: ContextRole;
  weight: ContextWeight;
  content: string;
  source: string;
  tokenCount: number;
  createdAt: number;
  referencedBy: UUID[];
}

export function makeContextBlock(
  role: ContextRole,
  weight: ContextWeight,
  content: string,
  source: string,
): ContextBlock {
  return {
    id: uuidv4(),
    role,
    weight,
    content,
    source,
    tokenCount: estimateTokens(content),
    createdAt: 0,
    referencedBy: [],
  };
}

export enum ModelTier {
  Cheap = 'cheap',
  Standard = 'standard',
  Premium = 'premium',
  Auto = 'auto',
}

export interface EvidenceRef {
  id: UUID;
  content: string;
  sourceUrl?: string;
  toolName?: string;
  confidence: number;
}

export interface DecisionRecord {
  id: UUID;
  decision: string;
  reasoning: string;
  evidenceIds: UUID[];
  confidence: number;
  decidedBy: string;
  timestamp: number;
}

export interface AgentEnvelope {
  id: UUID;
  task: string;
  context: ContextBlock[];
  evidence: EvidenceRef[];
  decisions: DecisionRecord[];
  budgetUsdRemaining: number | null;
  checkpointId?: UUID;
  modelHint: ModelTier;
  parentAgent?: string;
  scope: string[];
  createdAt: number;
}

export function makeEnvelope(task: string): AgentEnvelope {
  return {
    id: uuidv4(),
    task,
    context: [],
    evidence: [],
    decisions: [],
    budgetUsdRemaining: null,
    modelHint: ModelTier.Auto,
    scope: [],
    createdAt: 0,
  };
}
