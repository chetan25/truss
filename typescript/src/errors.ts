export class TrussError extends Error {
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

export class BudgetExceeded extends TrussError {}
export class ToolOutOfScope extends TrussError {}
export class CheckpointNotFound extends TrussError {}
export class FenceLockConflict extends TrussError {}
