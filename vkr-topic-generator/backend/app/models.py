from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Teacher(Base):
    __tablename__ = "teachers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    topic_count: Mapped[int] = mapped_column(Integer, nullable=False, default=5)

    department: Mapped[str | None] = mapped_column(String(255), nullable=True)
    position: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    telegram: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Короткий научный профиль. Прошлые одобренные ВКР остаются главным сигналом генерации.
    research_areas: Mapped[list] = mapped_column(JSON, default=list)
    orcid: Mapped[str | None] = mapped_column(String(100), nullable=True)
    scopus_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Скрытые legacy-поля оставлены только для совместимости со старыми БД v5-v7.
    # В интерфейсе и генерации они не используются, но их Python-default предотвращает
    # HTTP 500 при INSERT в старую PostgreSQL-схему, где эти JSON-колонки были NOT NULL.
    courses: Mapped[list] = mapped_column(JSON, default=list)
    technologies: Mapped[list] = mapped_column(JSON, default=list)
    publications: Mapped[list] = mapped_column(JSON, default=list)
    openalex_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    past_topics: Mapped[list["PastTopic"]] = relationship(
        back_populates="teacher", cascade="all, delete-orphan"
    )
    topics: Mapped[list["GeneratedTopic"]] = relationship(
        back_populates="teacher", cascade="all, delete-orphan"
    )

    @property
    def contact_text(self) -> str:
        return " · ".join(x for x in [self.email, self.telegram, self.phone] if x)


class PastTopic(Base):
    __tablename__ = "past_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    teacher_id: Mapped[int] = mapped_column(ForeignKey("teachers.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Происхождение делает массовый XLSX-импорт проверяемым и воспроизводимым.
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_sheet: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_row: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_project: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_reference: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    teacher: Mapped[Teacher] = relationship(back_populates="past_topics")


class GeneratedTopic(Base):
    __tablename__ = "generated_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    teacher_id: Mapped[int] = mapped_column(ForeignKey("teachers.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    generation_source: Mapped[str] = mapped_column(String(32), nullable=False, default="legacy")
    generation_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Зафиксированный результат профильного Quality Gate. Для старых/ручных
    # тем поля могут быть NULL и тогда score вычисляется на чтении.
    profile_relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_research_areas: Mapped[list] = mapped_column(JSON, default=list)
    profile_directions: Mapped[list] = mapped_column(JSON, default=list)
    foreign_profile_directions: Mapped[list] = mapped_column(JSON, default=list)
    profile_relevance_method: Mapped[str | None] = mapped_column(String(80), nullable=True)

    # similarity_score оставлен как совместимое поле и равен глобальной похожести.
    similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    closest_past_topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    teacher_similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    teacher_closest_topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    global_similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    global_closest_topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    global_closest_teacher: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Отдельный сигнал только по другим темам текущего набора. Он позволяет
    # отличить повтор внутри batch от совпадения с исторической базой.
    batch_similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    batch_closest_topic: Mapped[str | None] = mapped_column(Text, nullable=True)
    batch_closest_teacher: Mapped[str | None] = mapped_column(String(255), nullable=True)
    similarity_method: Mapped[str] = mapped_column(String(50), default="embeddings")
    teacher_similarity_method: Mapped[str] = mapped_column(String(50), default="none")
    global_similarity_method: Mapped[str] = mapped_column(String(50), default="none")
    batch_similarity_method: Mapped[str] = mapped_column(String(50), default="none")

    # Если тема в demo-режиме была перенесена дословно из списка ранее
    # одобренных тем преподавателя, здесь сохраняется конкретный источник.
    # Это позволяет исключить только эту исходную запись из проверки похожести
    # и не скрывать реальные дубликаты с другими прошлыми темами.
    source_past_topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("past_topics.id", ondelete="SET NULL"), nullable=True, index=True
    )

    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)
    manual_edit: Mapped[bool] = mapped_column(Boolean, default=False)
    generation_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    teacher: Mapped[Teacher] = relationship(back_populates="topics")


class GenerationBatch(Base):
    __tablename__ = "generation_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    focus: Mapped[str | None] = mapped_column(Text, nullable=True)
    selections: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    topic_links: Mapped[list["GenerationBatchTopic"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class GenerationBatchTopic(Base):
    __tablename__ = "generation_batch_topics"

    batch_id: Mapped[int] = mapped_column(
        ForeignKey("generation_batches.id", ondelete="CASCADE"), primary_key=True
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("generated_topics.id", ondelete="CASCADE"), primary_key=True, unique=True
    )

    batch: Mapped[GenerationBatch] = relationship(back_populates="topic_links")
    topic: Mapped[GeneratedTopic] = relationship()


class AppSettings(Base):
    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    mistral_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    mistral_chat_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    mistral_embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    google_service_account_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    google_share_with_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    google_spreadsheet_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    default_topic_count: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    default_generation_focus: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Фиксированная шкала гибридного индекса похожести.
    similarity_green_max: Mapped[float] = mapped_column(Float, nullable=False, default=40.0)
    similarity_yellow_max: Mapped[float] = mapped_column(Float, nullable=False, default=70.0)
    similarity_calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class QualityEvaluationRun(Base):
    __tablename__ = "quality_evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("generation_batches.id", ondelete="CASCADE"), index=True)
    reference_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    topic_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mode: Mapped[str] = mapped_column(String(50), nullable=False, default="heuristic")
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    items: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class SimilarityCalibrationPair(Base):
    __tablename__ = "similarity_calibration_pairs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    left_title: Mapped[str] = mapped_column(Text, nullable=False)
    right_title: Mapped[str] = mapped_column(Text, nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    similarity_method: Mapped[str] = mapped_column(String(50), nullable=False, default="lexical-fallback")
    label: Mapped[str | None] = mapped_column(String(20), nullable=True)  # different | similar | duplicate
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
