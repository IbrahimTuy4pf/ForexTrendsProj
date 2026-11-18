"""Walk-forward fold indices. Time order only; labels that overlap the test window are purged."""

from __future__ import annotations

from dataclasses import dataclass

from quant_forex.config.schema import WalkForwardConfig


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    mode: str
    fit_start: int
    fit_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int
    purge_gap: int

    def assert_order(self) -> None:
        if not (0 <= self.fit_start < self.fit_end <= self.val_start < self.val_end):
            raise ValueError("Fit/validation indices are not chronological.")
        if not (self.val_end <= self.test_start < self.test_end):
            raise ValueError("Validation must end before the test window.")
        if self.test_start - self.val_end < self.purge_gap:
            raise ValueError("Purge/embargo gap is smaller than configured.")


def generate_walk_forward_folds(
    n_rows: int,
    config: WalkForwardConfig,
    horizon: int,
) -> list[WalkForwardFold]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    purge_gap = horizon + config.embargo_bars
    folds: list[WalkForwardFold] = []
    cursor = config.train_bars
    fold_id = 0
    while True:
        test_start = cursor + purge_gap
        test_end = test_start + config.test_bars
        if test_end > n_rows:
            break
        val_end = cursor
        val_start = val_end - config.validation_bars
        if config.mode == "expanding":
            fit_start = 0
        else:
            fit_start = max(0, val_end - config.train_bars)
        fit_end = val_start
        if fit_end - fit_start < 10 or val_start <= fit_start:
            break
        fold = WalkForwardFold(
            fold_id=fold_id,
            mode=config.mode,
            fit_start=fit_start,
            fit_end=fit_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
            purge_gap=purge_gap,
        )
        fold.assert_order()
        folds.append(fold)
        cursor += config.step_bars
        fold_id += 1
    return folds
