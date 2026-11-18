"""Project root discovery so paths are never hard-coded to a machine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def find(cls, start: Path | None = None) -> ProjectPaths:
        current = (start or Path.cwd()).resolve()
        for candidate in [current, *current.parents]:
            if (candidate / "pyproject.toml").is_file():
                return cls(root=candidate)
        raise FileNotFoundError(
            "Could not locate project root (no pyproject.toml in parent directories)."
        )

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.root / candidate).resolve()

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def data_external(self) -> Path:
        return self.root / "data" / "external"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def configs(self) -> Path:
        return self.root / "configs"
