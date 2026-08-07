from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from ai_software_engineer_agent.domain.entities import AgentFeature, AgentRun


class SQLiteAgentRunRepository:
    def __init__(self, database_path: str) -> None:
        self._database_path = database_path
        if database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save(self, run: AgentRun) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (id, feature, prompt, context, response, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.feature.value,
                    run.prompt,
                    run.context,
                    run.response,
                    run.created_at.isoformat(),
                ),
            )

    def list_recent(self, limit: int = 20) -> list[AgentRun]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, feature, prompt, context, response, created_at
                FROM agent_runs
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            AgentRun(
                id=row["id"],
                feature=AgentFeature(row["feature"]),
                prompt=row["prompt"],
                context=row["context"],
                response=row["response"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    feature TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    context TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_runs_created_at
                ON agent_runs (created_at DESC)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        return connection
