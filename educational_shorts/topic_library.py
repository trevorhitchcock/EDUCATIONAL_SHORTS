from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from educational_shorts.schemas import VideoTopic


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_topic_library(database_path: Path) -> sqlite3.Connection:
    """Open the topic library and return rows by column name."""
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON;")
    connection.execute("PRAGMA journal_mode = WAL;")

    return connection


def initialize_topic_library(database_path: Path) -> None:
    """Create the topic-library schema when it does not yet exist."""
    with connect_topic_library(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                learning_objective TEXT NOT NULL,
                category_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'approved'
                    CHECK (
                        status IN (
                            'approved',
                            'processing',
                            'completed',
                            'failed',
                            'rejected'
                        )
                    ),
                source TEXT NOT NULL DEFAULT 'topic_library_notebook',
                created_at_utc TEXT NOT NULL,
                selected_at_utc TEXT,
                completed_at_utc TEXT,
                failure_message TEXT,
                final_video_path TEXT,
                UNIQUE(title)
            )
            """
        )


def add_topic(
    database_path: Path,
    topic: VideoTopic,
    category_path: list[str],
    status: str = "approved",
) -> int | None:
    """
    Add a topic to the library.

    Returns its database ID, or None when its title already exists.
    """
    initialize_topic_library(database_path)

    try:
        with connect_topic_library(database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO topics (
                    title,
                    learning_objective,
                    category_path,
                    status,
                    created_at_utc
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    topic.title,
                    topic.learning_objective,
                    json.dumps(category_path, ensure_ascii=False),
                    status,
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    except sqlite3.IntegrityError:
        return None


def claim_next_topic(
    database_path: Path,
    category_root: str | None = None,
) -> tuple[int, VideoTopic, list[str]] | None:
    """
    Atomically choose one approved topic and mark it as processing.

    Oldest approved topics are selected first.
    """
    initialize_topic_library(database_path)

    with connect_topic_library(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")

        if category_root is None:
            row = connection.execute(
                """
                SELECT id, title, learning_objective, category_path
                FROM topics
                WHERE status = 'approved'
                ORDER BY created_at_utc, id
                LIMIT 1
                """
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT id, title, learning_objective, category_path
                FROM topics
                WHERE status = 'approved'
                  AND json_extract(category_path, '$[0]') = ?
                ORDER BY created_at_utc, id
                LIMIT 1
                """,
                (category_root,),
            ).fetchone()

        if row is None:
            connection.rollback()
            return None

        connection.execute(
            """
            UPDATE topics
            SET status = 'processing',
                selected_at_utc = ?,
                failure_message = NULL
            WHERE id = ?
            """,
            (_utc_now(), row["id"]),
        )

        connection.commit()

        category_path = json.loads(row["category_path"])

        topic = VideoTopic(
            title=row["title"],
            category_path=category_path,
            learning_objective=row["learning_objective"],
        )

        return int(row["id"]), topic, category_path


def mark_topic_completed(
    database_path: Path,
    topic_id: int,
    final_video_path: Path,
) -> None:
    with connect_topic_library(database_path) as connection:
        connection.execute(
            """
            UPDATE topics
            SET status = 'completed',
                completed_at_utc = ?,
                final_video_path = ?,
                failure_message = NULL
            WHERE id = ?
            """,
            (_utc_now(), str(final_video_path), topic_id),
        )


def mark_topic_failed(
    database_path: Path,
    topic_id: int,
    message: str,
) -> None:
    with connect_topic_library(database_path) as connection:
        connection.execute(
            """
            UPDATE topics
            SET status = 'failed',
                failure_message = ?
            WHERE id = ?
            """,
            (message, topic_id),
        )


def reset_failed_topic(
    database_path: Path,
    topic_id: int,
) -> None:
    """Return a failed topic to the approved queue."""
    with connect_topic_library(database_path) as connection:
        connection.execute(
            """
            UPDATE topics
            SET status = 'approved',
                selected_at_utc = NULL,
                failure_message = NULL
            WHERE id = ?
            """,
            (topic_id,),
        )