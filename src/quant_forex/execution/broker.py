"""Order venues. Live routing is not implemented and cannot be enabled."""

from __future__ import annotations

from typing import Any


class LiveTradingNotImplemented(RuntimeError):
    """Raised if anything tries to send an order to a real broker."""


def submit_live_order(*_args: Any, **_kwargs: Any) -> None:
    raise LiveTradingNotImplemented(
        "Live trading is not implemented. This platform has no broker adapter "
        "and LIVE_TRADING cannot be enabled."
    )
