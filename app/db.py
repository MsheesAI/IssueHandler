"""Database models, engine, and persistence helpers."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.config import get_settings

log = logging.getLogger(__name__)

Base = declarative_base()


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    action = Column(String(50), nullable=False)  # create, close, comment
    repo = Column(String(200), nullable=False)
    issue_number = Column(Integer, nullable=True)
    payload = Column(Text, nullable=True)  # JSON-serialized details
    status = Column(String(50), default="success")


class PendingApproval(Base):
    __tablename__ = "pending_approvals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(100), unique=True, nullable=False, index=True)
    repo = Column(String(200), nullable=False)
    issue_number = Column(Integer, nullable=False)
    action_type = Column(String(50), default="close")
    summary = Column(Text, nullable=True)
    status = Column(String(20), default="pending")  # pending, approved, rejected
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


settings = get_settings()
engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

_initialized = False


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def record_audit(
    action: str,
    repo: str,
    issue_number: Optional[int] = None,
    payload: Optional[dict[str, Any]] = None,
    status: str = "success",
) -> None:
    """Write an audit row. Never raises: a logging problem must not break a tool call."""
    global _initialized
    try:
        if not _initialized:
            await init_db()
            _initialized = True
        async with AsyncSessionLocal() as session:
            session.add(
                AuditLog(
                    action=action,
                    repo=repo,
                    issue_number=issue_number,
                    payload=json.dumps(payload or {}, default=str),
                    status=status,
                )
            )
            await session.commit()
    except Exception as e:
        log.warning("Audit log failed (non-fatal): %s", e)
