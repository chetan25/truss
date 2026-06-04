from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from truss.types import ModelTier


class BudgetWindow(Enum):
    PER_SESSION = "per_session"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    PER_MONTH = "per_month"


@dataclass
class BudgetLimit:
    window: BudgetWindow
    tokens: Optional[int] = None
    usd: Optional[float] = None


@dataclass
class AlertConfig:
    slack_webhook: Optional[str] = None
    log_to_stderr: bool = True


class ExceededAction(Enum):
    BLOCK = "block"
    QUEUE = "queue"


@dataclass
class BudgetConfig:
    per_session: Optional[BudgetLimit] = None
    per_user: Optional[BudgetLimit] = None
    per_agent: Optional[BudgetLimit] = None
    global_limit: Optional[BudgetLimit] = None
    on_exceeded: ExceededAction = ExceededAction.BLOCK
    alert_at_pct: float = 0.8
    alerts: AlertConfig = field(default_factory=AlertConfig)
