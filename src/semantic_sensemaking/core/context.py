"""Per-run output directory and artifact bookkeeping.

Replaces the prototype's ``OUTPUT_DIR`` global and ``{stamp}``-suffixed file
names: each run gets one timestamped directory with stable file names inside,
which keeps downstream scripting and Tapis archiving simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunContext:
    run_dir: Path
    run_id: str
    workflow: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    artifacts: list[Path] = field(default_factory=list)

    @classmethod
    def create(cls, output_dir: str | Path, workflow: str, run_id: str | None = None) -> RunContext:
        """Create ``output_dir/run_id``. A generated id never reuses an existing directory."""
        output_dir = Path(output_dir)
        if run_id is None:
            base = f"{workflow}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_id, n = base, 2
            while (output_dir / run_id).exists():
                run_id, n = f"{base}_{n}", n + 1
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(run_dir=run_dir, run_id=run_id, workflow=workflow)

    def artifact(self, name: str) -> Path:
        """Return a path inside the run directory and record it as an artifact."""
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path not in self.artifacts:
            self.artifacts.append(path)
        return path

    def relative_artifacts(self) -> list[str]:
        return [p.relative_to(self.run_dir).as_posix() for p in self.artifacts]
