"""Pydantic configuration models.

All research assumptions live here (or in YAML), not as scattered constants.
Live trading cannot be enabled in this version.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    name: str = "quant-forex-ai"
    random_seed: int = 42
    log_level: str = "INFO"


class InstrumentConfig(BaseModel):
    symbol: str = "EURUSD"
    base_currency: str = "EUR"
    quote_currency: str = "USD"
    yfinance_ticker: str | None = None
    pip_size: float = 0.0001

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        cleaned = value.replace("/", "").replace("-", "").replace("_", "").strip().upper()
        if len(cleaned) != 6 or not cleaned.isalpha():
            raise ValueError(
                f"Symbol must be a 6-letter FX pair such as EURUSD, got {value!r}."
            )
        return cleaned

    @field_validator("base_currency", "quote_currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.strip().upper()

    def resolved_yfinance_ticker(self) -> str:
        if self.yfinance_ticker:
            return self.yfinance_ticker
        return f"{self.symbol}=X"


class DataConfig(BaseModel):
    source: Literal["csv", "yfinance"] = "csv"
    csv_path: str | None = "data/external/sample_eurusd_daily.csv"
    yfinance_period: str = "5y"
    yfinance_interval: str = "1d"
    timezone: str = "UTC"
    expected_frequency: str | None = "B"
    drop_duplicates: bool = True
    sort_chronologically: bool = True
    missing_value_policy: Literal["flag", "drop", "ffill"] = "flag"
    resample_rule: str | None = None
    abnormal_return_threshold: float = Field(default=0.05, gt=0)
    persist_raw: bool = True
    persist_processed: bool = True


class StorageConfig(BaseModel):
    sqlite_path: str = "data/processed/market_data.sqlite"


class CostConfig(BaseModel):
    spread_pips: float = 1.0
    slippage_pips: float = 0.2
    commission_per_lot: float = 0.0


class RiskConfig(BaseModel):
    """Conservative risk limits. None of these exist to maximize return."""

    risk_per_trade: float = Field(default=0.005, gt=0, le=0.05)
    max_risk_per_trade: float = Field(default=0.005, gt=0, le=0.05)
    max_daily_loss: float = Field(default=0.02, gt=0, le=0.5)
    max_drawdown: float = Field(default=0.10, gt=0, le=1.0)
    max_leverage: float = Field(default=5.0, ge=1.0)
    max_gross_exposure: float = Field(default=2.0, gt=0)
    max_concurrent_positions: int = Field(default=1, ge=1)
    max_position_size: float = Field(default=1.0, gt=0)
    sizing_method: Literal["fixed_fraction", "volatility", "min"] = "min"
    vol_lookback: int = Field(default=20, ge=5)
    annual_vol_target: float = Field(default=0.10, gt=0, le=1.0)
    periods_per_year: int = Field(default=252, ge=1)

    @model_validator(mode="after")
    def caps_at_least_as_tight_as_targets(self) -> RiskConfig:
        if self.max_risk_per_trade < self.risk_per_trade:
            raise ValueError("max_risk_per_trade must be >= risk_per_trade.")
        if self.max_gross_exposure > self.max_leverage:
            raise ValueError("max_gross_exposure cannot exceed max_leverage.")
        return self


class ExecutionConfig(BaseModel):
    """Execution switches. Live trading is hard-blocked."""

    live_trading: bool = False
    paper_trading: bool = False
    allow_live_override: bool = False

    @model_validator(mode="after")
    def forbid_live_trading(self) -> ExecutionConfig:
        if self.live_trading or self.allow_live_override:
            raise ValueError(
                "LIVE_TRADING is disabled in this research platform and cannot be "
                "activated from configuration. Broker execution is not implemented."
            )
        return self


class ExecutionEnvSettings(BaseSettings):
    """Environment-level guard. LIVE_TRADING=true is rejected."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    live_trading: bool = False

    @field_validator("live_trading")
    @classmethod
    def reject_live(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "LIVE_TRADING cannot be enabled. This project does not place real trades."
            )
        return False


class SplitConfig(BaseModel):
    train_end: str | None = "2021-12-31"
    validation_end: str | None = "2023-12-31"
    fallback_train_fraction: float = Field(default=0.60, gt=0, lt=1)
    fallback_validation_fraction: float = Field(default=0.20, gt=0, lt=1)

    @model_validator(mode="after")
    def fractions_fit(self) -> SplitConfig:
        if self.fallback_train_fraction + self.fallback_validation_fraction >= 0.95:
            raise ValueError("Leave a remainder for the out-of-sample test split.")
        return self


class ModelConfig(BaseModel):
    """Simple classifiers only. No test-set tuning. No deep learning in this phase."""

    compare: list[str] = Field(
        default_factory=lambda: ["logistic", "random_forest", "gradient_boosting"]
    )
    horizon: int = Field(default=1, ge=1, le=20)
    threshold: float = 0.0
    probability_long: float = Field(default=0.55, ge=0.5, le=1.0)
    probability_short: float = Field(default=0.45, ge=0.0, le=0.5)
    n_estimators: int = Field(default=80, ge=10)
    max_depth: int = Field(default=3, ge=1, le=8)
    min_samples_leaf: int = Field(default=20, ge=1)
    backtest_test_fold: bool = True

    @field_validator("compare")
    @classmethod
    def known_models(cls, value: list[str]) -> list[str]:
        allowed = {"logistic", "random_forest", "gradient_boosting"}
        unknown = [item for item in value if item not in allowed]
        if unknown:
            raise ValueError(f"Unknown models {unknown}. Allowed: {sorted(allowed)}")
        if not value:
            raise ValueError("models.compare must not be empty.")
        return value


class ExplorationConfig(BaseModel):
    rolling_window: int = Field(default=20, ge=2)
    autocorrelation_lags: int = Field(default=5, ge=1, le=20)
    save_matplotlib: bool = True
    save_plotly_html: bool = True


class FeatureConfig(BaseModel):
    """Which feature groups to build. Enabling a group does not imply it has an edge."""

    groups: list[str] = Field(
        default_factory=lambda: ["price", "trend", "momentum", "volatility", "time"]
    )
    sma_windows: list[int] = Field(default_factory=lambda: [10, 20, 50])
    ema_windows: list[int] = Field(default_factory=lambda: [12, 26])
    rolling_return_windows: list[int] = Field(default_factory=lambda: [5, 20])
    rsi_window: int = Field(default=14, ge=2)
    atr_window: int = Field(default=14, ge=2)
    roc_window: int = Field(default=10, ge=1)
    momentum_window: int = Field(default=10, ge=1)
    macd_fast: int = Field(default=12, ge=2)
    macd_slow: int = Field(default=26, ge=3)
    macd_signal: int = Field(default=9, ge=2)
    bb_window: int = Field(default=20, ge=2)
    bb_std: float = Field(default=2.0, gt=0)
    vol_window: int = Field(default=20, ge=2)
    vol_regime_window: int = Field(default=60, ge=5)
    drop_warmup_rows: bool = False
    attach_next_return_target: bool = False

    @field_validator("groups")
    @classmethod
    def known_groups(cls, value: list[str]) -> list[str]:
        allowed = {"price", "trend", "momentum", "volatility", "time"}
        unknown = [item for item in value if item not in allowed]
        if unknown:
            raise ValueError(f"Unknown feature groups {unknown}. Allowed: {sorted(allowed)}")
        if not value:
            raise ValueError("features.groups must not be empty.")
        return value

    @model_validator(mode="after")
    def macd_spans(self) -> FeatureConfig:
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be smaller than macd_slow.")
        return self


class StrategyParams(BaseModel):
    enabled: bool = True


class MACrossoverParams(StrategyParams):
    fast: int = Field(default=10, ge=2)
    slow: int = Field(default=20, ge=3)

    @model_validator(mode="after")
    def fast_slower_than_slow(self) -> MACrossoverParams:
        if self.fast >= self.slow:
            raise ValueError("ma_crossover.fast must be smaller than slow.")
        return self


class MomentumParams(StrategyParams):
    lookback: int = Field(default=20, ge=1)
    threshold: float = Field(default=0.0, ge=0.0)


class MeanReversionParams(StrategyParams):
    window: int = Field(default=20, ge=5)
    zscore: float = Field(default=1.0, gt=0)


class BreakoutParams(StrategyParams):
    lookback: int = Field(default=20, ge=2)


class BuyAndHoldParams(StrategyParams):
    pass


class StrategyConfig(BaseModel):
    allow_short: bool = True
    atr_window: int = Field(default=14, ge=2)
    stop_atr_multiple: float | None = Field(default=1.5, gt=0)
    take_profit_atr_multiple: float | None = Field(default=3.0, gt=0)
    buy_and_hold: BuyAndHoldParams = Field(default_factory=BuyAndHoldParams)
    ma_crossover: MACrossoverParams = Field(default_factory=MACrossoverParams)
    momentum: MomentumParams = Field(default_factory=MomentumParams)
    mean_reversion: MeanReversionParams = Field(default_factory=MeanReversionParams)
    breakout: BreakoutParams = Field(default_factory=BreakoutParams)


class BacktestConfig(BaseModel):
    initial_capital: float = Field(default=10_000.0, gt=0)
    lot_size: float = Field(default=100_000.0, gt=0)
    fill_on: Literal["next_open"] = "next_open"
    same_bar_stop_take: Literal["stop_first"] = "stop_first"
    account_currency: str = "USD"


class WalkForwardConfig(BaseModel):
    """Chronological expanding or rolling evaluation. Never shuffle."""

    mode: Literal["expanding", "rolling"] = "expanding"
    train_bars: int = Field(default=80, ge=20)
    validation_bars: int = Field(default=20, ge=5)
    test_bars: int = Field(default=20, ge=5)
    step_bars: int = Field(default=20, ge=1)
    embargo_bars: int = Field(default=0, ge=0)
    backtest_oos: bool = True

    @model_validator(mode="after")
    def train_has_room_for_validation(self) -> WalkForwardConfig:
        if self.validation_bars >= self.train_bars:
            raise ValueError("walk_forward.validation_bars must be smaller than train_bars.")
        return self


class DiagnosticsConfig(BaseModel):
    """Overfitting and statistical diagnostics. Not a green-light to trade."""

    n_bootstrap: int = Field(default=300, ge=20, le=5000)
    n_permutations: int = Field(default=40, ge=5, le=500)
    block_size: int = Field(default=10, ge=2)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    too_good_total_return: float = Field(default=0.50, gt=0)
    too_good_max_drawdown: float = Field(default=0.03, gt=0, lt=1)
    too_good_sharpe: float = Field(default=2.5, gt=0)
    auc_gap_warn: float = Field(default=0.15, gt=0, lt=1)
    ma_fast_offsets: list[int] = Field(default_factory=lambda: [-2, 0, 2])
    ma_slow_offsets: list[int] = Field(default_factory=lambda: [-4, 0, 4])
    momentum_lookback_offsets: list[int] = Field(default_factory=lambda: [-5, 0, 5])


class PaperConfig(BaseModel):
    """Simulated account only. Never contacts a broker."""

    ledger_path: str = "data/processed/paper_ledger.sqlite"
    strategy: str | None = None


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    timeframe: str = "1d"
    data: DataConfig = Field(default_factory=DataConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    strategies: StrategyConfig = Field(default_factory=StrategyConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    paper: PaperConfig = Field(default_factory=PaperConfig)
