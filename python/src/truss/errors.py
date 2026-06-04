class TrussError(Exception):
    pass

class BudgetExceeded(TrussError):
    pass

class ToolOutOfScope(TrussError):
    pass

class CheckpointNotFound(TrussError):
    pass

class FenceLockConflict(TrussError):
    pass
