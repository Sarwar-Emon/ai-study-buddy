"""Protects a public demo: one shared budget of model-using requests per day."""

import threading
from collections.abc import Callable
from datetime import date, datetime, timezone


def _today() -> date:
    return datetime.now(timezone.utc).date()


class DemoGuard:
    def __init__(self, daily_limit: int, today: Callable[[], date] = _today):
        self._limit = daily_limit
        self._today = today
        self._day = today()
        self._used = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        """Spend one request. False when today's budget is used up (it resets at midnight UTC)."""
        with self._lock:
            if self._today() != self._day:
                self._day, self._used = self._today(), 0
            if self._used >= self._limit:
                return False
            self._used += 1
            return True
