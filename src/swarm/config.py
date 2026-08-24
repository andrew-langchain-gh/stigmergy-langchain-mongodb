"""Runtime settings, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "inc-1042"
STATE_DIR = REPO_ROOT / ".swarm"

INCIDENT_ID = "INC-1042"


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str = field(
        default_factory=lambda: os.environ.get(
            "MONGODB_URI", "mongodb://swarm:swarm@localhost:27018/?directConnection=true"
        )
    )
    db_name: str = field(default_factory=lambda: os.environ.get("MONGODB_DB", "incident_swarm"))
    model: str = field(
        default_factory=lambda: os.environ.get("SWARM_MODEL", "anthropic:claude-sonnet-5")
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("SWARM_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    pace_ms: int = field(default_factory=lambda: int(os.environ.get("SWARM_PACE_MS", "2500")))
    max_cycles: int = field(default_factory=lambda: int(os.environ.get("SWARM_MAX_CYCLES", "12")))
    phase1_timeout: int = field(
        default_factory=lambda: int(os.environ.get("SWARM_PHASE1_TIMEOUT", "240"))
    )
    phase2_timeout: int = field(
        default_factory=lambda: int(os.environ.get("SWARM_PHASE2_TIMEOUT", "240"))
    )

    @property
    def embedding_dims(self) -> int:
        return 3072 if "large" in self.embedding_model else 1536


settings = Settings()
