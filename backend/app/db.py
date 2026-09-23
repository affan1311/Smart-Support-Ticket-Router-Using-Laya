"""Database connection. SQLite locally; set DATABASE_URL to a Postgres URL to switch."""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

# SQLite blocks cross-thread use by default, but background triage runs in a worker
# thread with its own session, so that check is turned off. Other databases don't need this.
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: one session per request, always closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    # Fine for a demo. A production app would use Alembic migrations instead.
    from app import models  # noqa: F401  (import registers the tables on Base)

    Base.metadata.create_all(bind=engine)
