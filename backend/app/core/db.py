from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    task: Mapped[str] = mapped_column(String(10))  # "stt" | "tts"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    datasets: Mapped[list["Dataset"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    runs: Mapped[list["Run"]] = relationship(back_populates="project", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String(200))
    task: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20), default="uploaded")  # uploaded|analyzed|prepared
    path: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(20), default="en")
    analysis: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    prepared: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    project: Mapped[Project] = relationship(back_populates="datasets")


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    model_id: Mapped[str] = mapped_column(String(100))
    method: Mapped[str] = mapped_column(String(30))
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    group: Mapped[str | None] = mapped_column(String(50), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued|running|completed|failed|stopped
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    train_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    peak_gpu_gb: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    project: Mapped[Project] = relationship(back_populates="runs")


class RegisteredModel(Base):
    __tablename__ = "registered_models"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    base_model_id: Mapped[str] = mapped_column(String(100))
    method: Mapped[str] = mapped_column(String(30))
    version: Mapped[int] = mapped_column(Integer, default=1)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), nullable=True)
    dataset_name: Mapped[str] = mapped_column(String(200), default="")
    path: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


engine = None
SessionLocal = None


def init_db(url: str | None = None) -> None:
    global engine, SessionLocal
    settings.ensure_dirs()
    engine = create_engine(url or settings.db_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)


def get_session():
    if SessionLocal is None:
        init_db()
    with SessionLocal() as session:
        yield session
