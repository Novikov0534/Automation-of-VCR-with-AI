from __future__ import annotations

from datetime import datetime
import asyncio
import json
import os
import re

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, case, delete as sa_delete, func, inspect, or_, select, text
from sqlalchemy.orm import Session, selectinload

from .config import get_settings
from .db import Base, engine, get_db
from .models import (
    AppSettings,
    GeneratedTopic,
    GenerationBatch,
    GenerationBatchTopic,
    PastTopic,
    QualityEvaluationRun,
    SimilarityCalibrationPair,
    Teacher,
)
from .runtime_settings import DISABLED_SECRET, load_runtime_settings, secret_hint
from .schemas import (
    AppSettingsRead,
    AppSettingsUpdate,
    CalibrationLabelUpdate,
    CalibrationPairRead,
    CalibrationSummary,
    GenerateRequest,
    GenerateSelectedRequest,
    GenerationBatchRead,
    GenerationResult,
    GenerationSelection,
    GoogleSheetResult,
    GoogleCheckRequest,
    MistralCheckRequest,
    HistoryDeleteResult,
    HistoryImportItem,
    HistoryImportPreview,
    HistoryImportResult,
    HistoryImportSheet,
    IntegrationsStatusRead,
    IntegrationCheckRead,
    PastTopicCreate,
    QualityEvaluationItem,
    QualityEvaluationRead,
    TeacherCreate,
    TeacherImportPayload,
    TeacherImportResult,
    TeacherRead,
    TeacherUpdate,
    TopicRead,
    TopicStatusUpdate,
    TopicUpdate,
)
from .services.calibration import apply_recommended_thresholds, generate_calibration_pairs, recommend_thresholds
from .services.evaluation import reference_quality_score, refresh_reference_set, run_quality_evaluation
from .services.generator import TopicGenerator
from .services.integration_checks import check_mistral, check_google
from .services.research_taxonomy import RESEARCH_AREA_NAMES
from .services.similarity import SimilarityService, SIMILARITY_HIGH_THRESHOLD, SIMILARITY_REVIEW_THRESHOLD
from .services.xlsx_exporter import build_topics_xlsx
from .services.xlsx_importer import DetectedTopic, detect_historical_topics
from .utils import teacher_table_name

settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


def _add_column_if_missing(table_name: str, column_name: str, ddl: str) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns(table_name)}
    if column_name not in columns:
        with engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl}"))


def ensure_compatible_columns() -> None:
    """Мягкое обновление старой базы без удаления преподавателей и тем."""
    for name, ddl in [
        ("scopus_id", "VARCHAR(100)"),
    ]:
        _add_column_if_missing("teachers", name, ddl)

    for name, ddl in [
        ("source_file", "VARCHAR(255)"),
        ("source_sheet", "VARCHAR(255)"),
        ("source_row", "INTEGER"),
        ("source_project", "VARCHAR(100)"),
        ("is_reference", "BOOLEAN DEFAULT FALSE"),
    ]:
        _add_column_if_missing("past_topics", name, ddl)

    for name, ddl in [
        ("teacher_similarity_score", "FLOAT DEFAULT 0"),
        ("teacher_closest_topic", "TEXT"),
        ("global_similarity_score", "FLOAT DEFAULT 0"),
        ("global_closest_topic", "TEXT"),
        ("global_closest_teacher", "VARCHAR(255)"),
        ("source_past_topic_id", "INTEGER"),
        ("teacher_similarity_method", "VARCHAR(50) DEFAULT 'none'"),
        ("global_similarity_method", "VARCHAR(50) DEFAULT 'none'"),
    ]:
        _add_column_if_missing("generated_topics", name, ddl)

    for name, ddl in [
        ("mistral_api_key", "TEXT"),
        ("mistral_chat_model", "VARCHAR(255)"),
        ("mistral_embedding_model", "VARCHAR(255)"),
        ("similarity_green_max", "FLOAT DEFAULT 40"),
        ("similarity_yellow_max", "FLOAT DEFAULT 70"),
        ("similarity_calibrated", "BOOLEAN DEFAULT FALSE"),
    ]:
        _add_column_if_missing("app_settings", name, ddl)


ensure_compatible_columns()


def clear_removed_ai_provider_secrets() -> None:
    """Удаляет секреты AI-провайдеров, которые больше не используются в v23.

    Старые физические колонки могут остаться в PostgreSQL после обновления базы,
    но их ключи не должны храниться без необходимости. Сами колонки не удаляем,
    чтобы мягкое обновление оставалось совместимым со старыми volume.
    """
    existing = {column["name"] for column in inspect(engine).get_columns("app_settings")}
    legacy_secret_columns = [
        name for name in ("gemini_api_key", "openrouter_api_key")
        if name in existing
    ]
    if not legacy_secret_columns:
        return
    assignments = ", ".join(f"{name} = NULL" for name in legacy_secret_columns)
    with engine.begin() as connection:
        connection.execute(text(f"UPDATE app_settings SET {assignments}"))


clear_removed_ai_provider_secrets()


def sanitize_legacy_invalid_settings() -> None:
    """Убирает только заведомо невалидные тестовые секреты старых версий.

    Валидный, но неверный токен не трогаем: он должен отображаться красным.
    Очищаем только значения, которые технически невозможно отправить сервису:
    невалидный Mistral key и несуществующий путь вместо JSON Google.
    """
    with Session(engine) as db:
        row = db.get(AppSettings, 1)
        if not row:
            return
        changed = False
        mistral_key = (getattr(row, "mistral_api_key", None) or "").strip()
        if mistral_key and mistral_key != DISABLED_SECRET:
            try:
                mistral_key.encode("ascii")
                bad_mistral_key = any(ch.isspace() for ch in mistral_key)
            except UnicodeEncodeError:
                bad_mistral_key = True
            if bad_mistral_key:
                row.mistral_api_key = DISABLED_SECRET
                changed = True


        google_raw = (row.google_service_account_json or "").strip()
        if google_raw and google_raw != DISABLED_SECRET and not google_raw.startswith("{") and not os.path.isfile(google_raw):
            # UI принимает JSON. Старые тестовые строки вроде «авававава» не должны
            # навсегда держать красный индикатор после обновления.
            row.google_service_account_json = DISABLED_SECRET
            changed = True

        if changed:
            db.commit()


sanitize_legacy_invalid_settings()


def teacher_query():
    return select(Teacher).options(selectinload(Teacher.past_topics), selectinload(Teacher.topics))


def teacher_ordering():
    normalized_position = func.lower(func.trim(Teacher.position))
    position_rank = case(
        (normalized_position == "профессор", 0),
        (normalized_position == "доцент", 1),
        (normalized_position == "преподаватель", 2),
        else_=3,
    )
    return position_rank, Teacher.full_name


def topic_query():
    return select(GeneratedTopic).options(selectinload(GeneratedTopic.teacher))


def topic_ordering():
    normalized_position = func.lower(func.trim(Teacher.position))
    position_rank = case(
        (normalized_position == "профессор", 0),
        (normalized_position == "доцент", 1),
        (normalized_position == "преподаватель", 2),
        else_=3,
    )
    return position_rank, Teacher.full_name, GeneratedTopic.generation_index, GeneratedTopic.id


def topic_to_schema(topic: GeneratedTopic) -> TopicRead:
    global_score = float(topic.global_similarity_score or topic.similarity_score or 0.0)
    global_closest = topic.global_closest_topic or topic.closest_past_topic
    return TopicRead(
        id=topic.id,
        teacher_id=topic.teacher_id,
        teacher_name=teacher_table_name(topic.teacher.full_name, topic.teacher.position),
        teacher_contact=topic.teacher.contact_text,
        title=topic.title,
        rationale=topic.rationale,
        keywords=topic.keywords or [],
        similarity_score=round(global_score, 1),
        closest_past_topic=global_closest,
        teacher_similarity_score=round(float(topic.teacher_similarity_score or 0.0), 1),
        teacher_closest_topic=topic.teacher_closest_topic,
        global_similarity_score=round(global_score, 1),
        global_closest_topic=global_closest,
        global_closest_teacher=topic.global_closest_teacher,
        similarity_method=topic.similarity_method,
        teacher_similarity_method=getattr(topic, "teacher_similarity_method", None) or "none",
        global_similarity_method=getattr(topic, "global_similarity_method", None) or topic.similarity_method or "none",
        status=topic.status,
        manual_edit=topic.manual_edit,
        created_at=topic.created_at,
        updated_at=topic.updated_at,
    )


def batch_to_schema(batch: GenerationBatch, db: Session) -> GenerationBatchRead:
    runtime = load_runtime_settings(db)
    base = (
        select(GeneratedTopic)
        .join(GenerationBatchTopic, GenerationBatchTopic.topic_id == GeneratedTopic.id)
        .where(GenerationBatchTopic.batch_id == batch.id)
    )
    topic_count = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    approved_count = db.scalar(select(func.count()).select_from(base.where(GeneratedTopic.status == "approved").subquery())) or 0
    # В v19 используется фиксированный гибридный индекс, а не raw cosine внешней embedding-модели.
    # Старые методы v18 не учитываем как «красные», пока набор не будет пересчитан.
    attention_condition = or_(
        GeneratedTopic.global_similarity_method == "exact-duplicate",
        GeneratedTopic.similarity_method == "exact-duplicate",
        and_(
            GeneratedTopic.global_similarity_method.in_(["hybrid-mistral", "hybrid-lexical"]),
            func.coalesce(GeneratedTopic.global_similarity_score, GeneratedTopic.similarity_score) >= SIMILARITY_HIGH_THRESHOLD,
        ),
    )
    attention_count = db.scalar(
        select(func.count()).select_from(
            base.where(GeneratedTopic.status != "rejected", attention_condition).subquery()
        )
    ) or 0
    selections = [GenerationSelection(**item) for item in (batch.selections or [])]
    return GenerationBatchRead(
        id=batch.id,
        focus=batch.focus,
        selections=selections,
        topic_count=int(topic_count),
        teacher_count=len(selections),
        approved_count=int(approved_count),
        attention_count=int(attention_count),
        created_at=batch.created_at,
    )


@app.get("/api/research-areas")
def research_areas():
    """Канонические подсказки для научного профиля. Свои значения тоже разрешены."""
    return {"items": list(RESEARCH_AREA_NAMES)}


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    runtime = load_runtime_settings(db)
    return {
        "ok": True,
        "mistral_configured": bool(runtime.mistral_api_key),
        "google_sheets_configured": bool(runtime.google_service_account_json),
        "demo_mode": runtime.demo_mode,
    }


def _check_schema(result) -> IntegrationCheckRead:
    return IntegrationCheckRead(status=result.status, message=result.message, detail=result.detail)


@app.get("/api/integrations/status", response_model=IntegrationsStatusRead)
async def integration_status(db: Session = Depends(get_db)):
    runtime = load_runtime_settings(db)
    mistral_result, google_result = await asyncio.gather(
        check_mistral(
            runtime.mistral_api_key,
            chat_model=runtime.mistral_chat_model,
            embedding_model=runtime.mistral_embedding_model,
        ),
        check_google(runtime.google_service_account_json, runtime.google_spreadsheet_id),
        return_exceptions=True,
    )

    from .services.integration_checks import IntegrationCheckResult
    if isinstance(mistral_result, Exception):
        mistral_result = IntegrationCheckResult(
            "error", "ошибка подключения", f"Mistral API: {type(mistral_result).__name__}: {mistral_result}"
        )
    if isinstance(google_result, Exception):
        google_result = IntegrationCheckResult(
            "error", "ошибка подключения", f"Google Sheets: {type(google_result).__name__}: {google_result}"
        )

    return IntegrationsStatusRead(
        mistral=_check_schema(mistral_result),
        google_sheets=_check_schema(google_result),
    )


@app.post("/api/integrations/check/mistral", response_model=IntegrationCheckRead)
async def check_mistral_endpoint(payload: MistralCheckRequest, db: Session = Depends(get_db)):
    runtime = load_runtime_settings(db)
    result = await check_mistral(
        payload.api_key.strip() if payload.api_key and payload.api_key.strip() else runtime.mistral_api_key,
        chat_model=(payload.chat_model.strip() if payload.chat_model and payload.chat_model.strip() else runtime.mistral_chat_model),
        embedding_model=(payload.embedding_model.strip() if payload.embedding_model and payload.embedding_model.strip() else runtime.mistral_embedding_model),
        live_probe=True,
    )
    return _check_schema(result)


@app.post("/api/integrations/check/google", response_model=IntegrationCheckRead)
async def check_google_endpoint(payload: GoogleCheckRequest, db: Session = Depends(get_db)):
    runtime = load_runtime_settings(db)
    raw = payload.service_account_json.strip() if payload.service_account_json and payload.service_account_json.strip() else runtime.google_service_account_json
    spreadsheet_id = payload.spreadsheet_id.strip() if payload.spreadsheet_id and payload.spreadsheet_id.strip() else runtime.google_spreadsheet_id
    result = await check_google(raw, spreadsheet_id)
    return _check_schema(result)


def settings_to_schema(db: Session) -> AppSettingsRead:
    runtime = load_runtime_settings(db)
    return AppSettingsRead(
        mistral_configured=bool(runtime.mistral_api_key),
        mistral_key_hint=secret_hint(runtime.mistral_api_key, label="ключ"),
        mistral_chat_model=runtime.mistral_chat_model,
        mistral_embedding_model=runtime.mistral_embedding_model,
        google_service_account_configured=bool(runtime.google_service_account_json),
        google_service_account_hint=("JSON настроен" if runtime.google_service_account_json else None),
        google_share_with_email=runtime.google_share_with_email,
        google_spreadsheet_id=runtime.google_spreadsheet_id,
        default_topic_count=runtime.default_topic_count,
        default_generation_focus=runtime.default_generation_focus,
        similarity_green_max=SIMILARITY_REVIEW_THRESHOLD,
        similarity_yellow_max=SIMILARITY_HIGH_THRESHOLD,
        similarity_calibrated=True,
    )


@app.get("/api/settings", response_model=AppSettingsRead)
def read_app_settings(db: Session = Depends(get_db)):
    return settings_to_schema(db)


@app.put("/api/settings", response_model=AppSettingsRead)
def update_app_settings(payload: AppSettingsUpdate, db: Session = Depends(get_db)):
    row = db.get(AppSettings, 1)
    if not row:
        row = AppSettings(id=1)
        db.add(row)
        db.flush()
    values = payload.model_dump(exclude_unset=True)
    current_green = float(values.get("similarity_green_max") if values.get("similarity_green_max") is not None else (row.similarity_green_max if row.similarity_green_max is not None else 40))
    current_yellow = float(values.get("similarity_yellow_max") if values.get("similarity_yellow_max") is not None else (row.similarity_yellow_max if row.similarity_yellow_max is not None else 70))
    if current_green >= current_yellow:
        raise HTTPException(400, "Нижняя граница должна быть меньше верхней границы")

    old_embedding_model = (row.mistral_embedding_model or "").strip()
    new_embedding_model = values.get("mistral_embedding_model")
    embedding_changed = bool(new_embedding_model is not None and str(new_embedding_model).strip() and str(new_embedding_model).strip() != old_embedding_model)
    # Калибровка относится к конкретной конфигурации embeddings. При смене
    # модели или учётных данных не переносим старую шкалу автоматически.
    mistral_credentials_changed = "mistral_api_key" in values

    for key, value in values.items():
        if key in {"mistral_api_key", "google_service_account_json"} and value is None:
            value = DISABLED_SECRET
        elif isinstance(value, str):
            value = value.strip() or None
        setattr(row, key, value)

    if embedding_changed or mistral_credentials_changed:
        row.similarity_calibrated = False
        db.execute(sa_delete(SimilarityCalibrationPair))

    db.commit()
    return settings_to_schema(db)


@app.get("/api/teachers", response_model=list[TeacherRead])
def list_teachers(db: Session = Depends(get_db)):
    rank, name = teacher_ordering()
    return db.scalars(teacher_query().order_by(rank, name)).unique().all()


def _teacher_backup_entry(teacher: Teacher) -> dict:
    return {
        "full_name": teacher.full_name,
        "topic_count": teacher.topic_count or 5,
        "department": teacher.department,
        "position": teacher.position,
        "email": teacher.email,
        "telegram": teacher.telegram,
        "phone": teacher.phone,
        "research_areas": teacher.research_areas or [],
        "orcid": teacher.orcid,
        "scopus_id": teacher.scopus_id,
        "past_topics": [
            {
                "title": item.title,
                "year": item.year,
                "source_file": item.source_file,
                "source_sheet": item.source_sheet,
                "source_row": item.source_row,
                "source_project": item.source_project,
                "is_reference": bool(item.is_reference),
            }
            for item in sorted(teacher.past_topics, key=lambda x: (x.year or 0, x.id))
        ],
    }


@app.get("/api/teachers/export-json")
def export_teachers_json(db: Session = Depends(get_db)):
    rank, name = teacher_ordering()
    teachers = db.scalars(teacher_query().order_by(rank, name)).unique().all()
    payload = {
        "format": "vkr-ai-teachers",
        "version": 2,
        "exported_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "teachers": [_teacher_backup_entry(teacher) for teacher in teachers],
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    filename = f"vkr_ai_teachers_{datetime.utcnow().strftime('%Y-%m-%d')}.json"
    return Response(
        content=content,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/teachers/import-json", response_model=TeacherImportResult)
def import_teachers_json(payload: TeacherImportPayload, db: Session = Depends(get_db)):
    existing = db.scalars(teacher_query()).unique().all()
    by_name = {" ".join(teacher.full_name.split()).casefold(): teacher for teacher in existing}
    created = updated = 0
    for item in payload.teachers:
        normalized_name = " ".join(item.full_name.split())
        key = normalized_name.casefold()
        data = item.model_dump(exclude={"past_topics"})
        data["full_name"] = normalized_name
        teacher = by_name.get(key)
        if teacher is None:
            teacher = Teacher(**data)
            db.add(teacher)
            db.flush()
            by_name[key] = teacher
            created += 1
        else:
            for field, value in data.items():
                setattr(teacher, field, value)
            teacher.past_topics.clear()
            db.flush()
            updated += 1
        for past in item.past_topics:
            teacher.past_topics.append(PastTopic(**past.model_dump()))
    db.commit()
    refresh_reference_set(db)
    total = db.scalar(select(func.count()).select_from(Teacher)) or 0
    return TeacherImportResult(created=created, updated=updated, total=int(total))


@app.post("/api/teachers", response_model=TeacherRead, status_code=201)
def create_teacher(payload: TeacherCreate, db: Session = Depends(get_db)):
    data = payload.model_dump(exclude={"past_topics"})
    teacher = Teacher(**data)
    for item in payload.past_topics:
        teacher.past_topics.append(PastTopic(**item.model_dump()))
    db.add(teacher)
    db.commit()
    return db.scalars(teacher_query().where(Teacher.id == teacher.id)).unique().one()


@app.put("/api/teachers/{teacher_id}", response_model=TeacherRead)
def update_teacher(teacher_id: int, payload: TeacherUpdate, db: Session = Depends(get_db)):
    teacher = db.get(Teacher, teacher_id)
    if not teacher:
        raise HTTPException(404, "Преподаватель не найден")
    data = payload.model_dump(exclude={"past_topics"})
    for key, value in data.items():
        setattr(teacher, key, value)
    if payload.past_topics is not None:
        # Сохраняем provenance импортированных тем, если пользователь не менял само название.
        existing_by_title = {
            re.sub(r"\s+", " ", item.title.casefold()).strip(): item
            for item in teacher.past_topics
        }
        desired_keys: set[str] = set()
        for item in payload.past_topics:
            key = re.sub(r"\s+", " ", item.title.casefold()).strip()
            desired_keys.add(key)
            existing_topic = existing_by_title.get(key)
            if existing_topic is not None:
                if item.year is not None:
                    existing_topic.year = item.year
            else:
                teacher.past_topics.append(PastTopic(**item.model_dump()))
        for existing_topic in list(teacher.past_topics):
            key = re.sub(r"\s+", " ", existing_topic.title.casefold()).strip()
            if key not in desired_keys:
                db.delete(existing_topic)
    db.commit()
    refresh_reference_set(db)
    return db.scalars(teacher_query().where(Teacher.id == teacher_id)).unique().one()


@app.delete("/api/teachers/{teacher_id}", status_code=204)
def delete_teacher(teacher_id: int, db: Session = Depends(get_db)):
    teacher = db.get(Teacher, teacher_id)
    if not teacher:
        raise HTTPException(404, "Преподаватель не найден")
    db.delete(teacher)
    db.commit()
    return Response(status_code=204)


@app.post("/api/teachers/{teacher_id}/past-topics", response_model=TeacherRead)
def add_past_topic(teacher_id: int, payload: PastTopicCreate, db: Session = Depends(get_db)):
    teacher = db.get(Teacher, teacher_id)
    if not teacher:
        raise HTTPException(404, "Преподаватель не найден")
    db.add(PastTopic(teacher_id=teacher_id, **payload.model_dump()))
    db.commit()
    refresh_reference_set(db)
    return db.scalars(teacher_query().where(Teacher.id == teacher_id)).unique().one()


@app.delete("/api/past-topics/{past_topic_id}", status_code=204)
def delete_past_topic(past_topic_id: int, db: Session = Depends(get_db)):
    item = db.get(PastTopic, past_topic_id)
    if not item:
        raise HTTPException(404, "Тема не найдена")
    db.delete(item)
    db.commit()
    refresh_reference_set(db)
    return Response(status_code=204)


# ------------------------- XLSX history import -------------------------

def _name_tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё-]+", value.casefold().replace("ё", "е"))


def _initial_key(value: str) -> str:
    tokens = _name_tokens(value)
    if not tokens:
        return ""
    surname = tokens[0]
    initials = "".join(token[0] for token in tokens[1:3] if token)
    return surname + ":" + initials


def _full_key(value: str) -> str:
    return " ".join(_name_tokens(value))


def _can_create_teacher(name: str) -> bool:
    tokens = _name_tokens(name)
    return len(tokens) >= 3 and all(len(x) > 1 for x in tokens[:3])


def _match_import_item(item: DetectedTopic, teachers: list[Teacher]) -> tuple[Teacher | None, str]:
    full = _full_key(item.teacher_name)
    by_full = {_full_key(t.full_name): t for t in teachers}
    if full in by_full:
        return by_full[full], "exact"
    initials = _initial_key(item.teacher_name)
    matches = [t for t in teachers if _initial_key(t.full_name) == initials]
    if len(matches) == 1:
        return matches[0], "initials"
    return None, "unknown"


def _preview_history(content: bytes, filename: str, db: Session) -> HistoryImportPreview:
    sheets = detect_historical_topics(content, filename)
    teachers = db.scalars(teacher_query()).unique().all()
    existing = {
        (p.teacher_id, re.sub(r"\s+", " ", p.title.casefold()).strip())
        for p in db.scalars(select(PastTopic)).all()
    }
    items: list[HistoryImportItem] = []
    sheet_stats: list[HistoryImportSheet] = []
    matched = unknown = duplicates = recommended = 0
    for sheet in sheets:
        s_matched = s_unknown = 0
        for raw in sheet.items:
            teacher, match_type = _match_import_item(raw, teachers)
            normalized_title = re.sub(r"\s+", " ", raw.title.casefold()).strip()
            already = bool(teacher and (teacher.id, normalized_title) in existing)
            is_ref = reference_quality_score(raw.title) >= 62
            if teacher:
                matched += 1
                s_matched += 1
            else:
                unknown += 1
                s_unknown += 1
            if already:
                duplicates += 1
            if is_ref:
                recommended += 1
            items.append(HistoryImportItem(
                sheet=raw.sheet,
                row=raw.row,
                teacher_name=raw.teacher_name,
                title=raw.title,
                year=raw.year,
                project=raw.project,
                matched_teacher_id=teacher.id if teacher else None,
                matched_teacher_name=teacher.full_name if teacher else None,
                match_type=match_type,
                already_exists=already,
                recommended_reference=is_ref,
            ))
        sheet_stats.append(HistoryImportSheet(
            name=sheet.name,
            detected_rows=len(sheet.items),
            matched_rows=s_matched,
            unknown_rows=s_unknown,
        ))
    return HistoryImportPreview(
        filename=filename,
        total_detected=len(items),
        matched=matched,
        unknown=unknown,
        duplicates=duplicates,
        recommended_reference=recommended,
        sheets=sheet_stats,
        items=items,
    )


@app.post("/api/history-import/preview", response_model=HistoryImportPreview)
async def preview_history_xlsx(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(400, "Выберите файл XLSX")
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(413, "XLSX слишком большой (максимум 20 МБ)")
    try:
        return _preview_history(content, file.filename, db)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/history-import/confirm", response_model=HistoryImportResult)
async def confirm_history_xlsx(
    file: UploadFile = File(...),
    selected_sheets: str = Form("[]"),
    create_missing_teachers: bool = Form(False),
    refresh_reference: bool = Form(True),
    db: Session = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(400, "Выберите файл XLSX")
    try:
        sheet_filter = set(json.loads(selected_sheets))
    except Exception as exc:
        raise HTTPException(400, "Некорректный список листов") from exc
    content = await file.read()
    detected = detect_historical_topics(content, file.filename)
    teachers = db.scalars(teacher_query()).unique().all()
    existing_keys = {
        (p.teacher_id, re.sub(r"\s+", " ", p.title.casefold()).strip())
        for p in db.scalars(select(PastTopic)).all()
    }
    imported = skipped_duplicates = skipped_unknown = created_teachers = 0
    created_by_key: dict[str, Teacher] = {}

    for sheet in detected:
        if sheet_filter and sheet.name not in sheet_filter:
            continue
        for raw in sheet.items:
            teacher, _ = _match_import_item(raw, teachers + list(created_by_key.values()))
            if teacher is None and create_missing_teachers and _can_create_teacher(raw.teacher_name):
                key = _full_key(raw.teacher_name)
                teacher = created_by_key.get(key)
                if teacher is None:
                    teacher = Teacher(full_name=" ".join(raw.teacher_name.split()), topic_count=5)
                    db.add(teacher)
                    db.flush()
                    created_by_key[key] = teacher
                    created_teachers += 1
            if teacher is None:
                skipped_unknown += 1
                continue
            normalized_title = re.sub(r"\s+", " ", raw.title.casefold()).strip()
            key = (teacher.id, normalized_title)
            if key in existing_keys:
                skipped_duplicates += 1
                continue
            db.add(PastTopic(
                teacher_id=teacher.id,
                title=raw.title,
                year=raw.year,
                source_file=file.filename,
                source_sheet=raw.sheet,
                source_row=raw.row,
                source_project=raw.project,
                is_reference=False,
            ))
            existing_keys.add(key)
            imported += 1
    db.commit()
    reference_count = refresh_reference_set(db, limit=100) if refresh_reference else int(
        db.scalar(select(func.count()).select_from(PastTopic).where(PastTopic.is_reference.is_(True))) or 0
    )
    return HistoryImportResult(
        imported=imported,
        skipped_duplicates=skipped_duplicates,
        skipped_unknown=skipped_unknown,
        created_teachers=created_teachers,
        reference_count=reference_count,
    )


# ------------------------- generation history -------------------------

@app.get("/api/generation-batches/latest", response_model=GenerationBatchRead | None)
def latest_generation_batch(db: Session = Depends(get_db)):
    batch = db.scalars(select(GenerationBatch).order_by(GenerationBatch.id.desc()).limit(1)).one_or_none()
    return batch_to_schema(batch, db) if batch else None


@app.get("/api/generation-batches", response_model=list[GenerationBatchRead])
def list_generation_batches(db: Session = Depends(get_db)):
    batches = db.scalars(select(GenerationBatch).order_by(GenerationBatch.id.desc()).limit(100)).all()
    return [batch_to_schema(batch, db) for batch in batches]


def _delete_generation_batch_row(batch: GenerationBatch, db: Session) -> int:
    topic_ids = list(db.scalars(
        select(GenerationBatchTopic.topic_id).where(GenerationBatchTopic.batch_id == batch.id)
    ).all())
    db.execute(sa_delete(QualityEvaluationRun).where(QualityEvaluationRun.batch_id == batch.id))
    db.execute(sa_delete(GenerationBatchTopic).where(GenerationBatchTopic.batch_id == batch.id))
    if topic_ids:
        db.execute(sa_delete(GeneratedTopic).where(GeneratedTopic.id.in_(topic_ids)))
    db.delete(batch)
    return len(topic_ids)


@app.delete("/api/generation-batches/{batch_id}", response_model=HistoryDeleteResult)
def delete_generation_batch(batch_id: int, db: Session = Depends(get_db)):
    batch = db.get(GenerationBatch, batch_id)
    if not batch:
        raise HTTPException(404, "Набор не найден")
    deleted_topics = _delete_generation_batch_row(batch, db)
    db.commit()
    return HistoryDeleteResult(deleted_batches=1, deleted_topics=deleted_topics)


@app.delete("/api/generation-batches", response_model=HistoryDeleteResult)
def clear_generation_history(db: Session = Depends(get_db)):
    batches = db.scalars(select(GenerationBatch).order_by(GenerationBatch.id.desc())).all()
    deleted_topics = 0
    for batch in batches:
        deleted_topics += _delete_generation_batch_row(batch, db)
    db.flush()
    # Полная очистка означает новый цикл нумерации наборов. В PostgreSQL id
    # использует sequence, поэтому сбрасываем её явно; SQLite после пустой таблицы
    # и INTEGER PRIMARY KEY сам начнёт с 1.
    if engine.dialect.name == "postgresql":
        db.execute(text("SELECT setval(pg_get_serial_sequence('generation_batches','id'), 1, false)"))
    elif engine.dialect.name == "sqlite":
        try:
            db.execute(text("DELETE FROM sqlite_sequence WHERE name='generation_batches'"))
        except Exception:
            pass
    db.commit()
    return HistoryDeleteResult(deleted_batches=len(batches), deleted_topics=deleted_topics)


@app.post("/api/generation-batches/{batch_id}/recalculate-similarity", response_model=list[TopicRead])
async def recalculate_generation_batch_similarity(batch_id: int, db: Session = Depends(get_db)):
    if not db.get(GenerationBatch, batch_id):
        raise HTTPException(404, "Набор не найден")
    await _recalculate_batch_similarities(batch_id, db)
    rank, name, generation_index, topic_id = topic_ordering()
    query = (
        topic_query()
        .join(Teacher, Teacher.id == GeneratedTopic.teacher_id)
        .join(GenerationBatchTopic, GenerationBatchTopic.topic_id == GeneratedTopic.id)
        .where(GenerationBatchTopic.batch_id == batch_id)
        .order_by(rank, name, generation_index, topic_id)
    )
    return [topic_to_schema(t) for t in db.scalars(query).unique().all()]


@app.get("/api/topics", response_model=list[TopicRead])
def list_topics(status: str | None = None, batch_id: int | None = None, db: Session = Depends(get_db)):
    rank, name, generation_index, topic_id = topic_ordering()
    query = topic_query().join(Teacher, Teacher.id == GeneratedTopic.teacher_id)
    if batch_id is not None:
        query = query.join(GenerationBatchTopic, GenerationBatchTopic.topic_id == GeneratedTopic.id).where(
            GenerationBatchTopic.batch_id == batch_id
        )
    if status:
        query = query.where(GeneratedTopic.status == status)
    query = query.order_by(rank, name, generation_index, topic_id)
    return [topic_to_schema(t) for t in db.scalars(query).unique().all()]


async def _recalculate_batch_similarities(batch_id: int, db: Session) -> None:
    current = db.scalars(
        select(GeneratedTopic)
        .join(GenerationBatchTopic, GenerationBatchTopic.topic_id == GeneratedTopic.id)
        .where(GenerationBatchTopic.batch_id == batch_id, GeneratedTopic.status != "rejected")
        .options(selectinload(GeneratedTopic.teacher))
        .order_by(GeneratedTopic.id)
    ).unique().all()
    if not current:
        return
    current_ids = {t.id for t in current}

    # Для PastTopic сохраняем ID конкретной исторической записи. Если текущая тема
    # была перенесена из неё дословно, исключаем только этот source_past_topic_id.
    # Другие одинаковые темы (у этого же или другого преподавателя) остаются в
    # сравнении и по-прежнему честно дают 100%.
    past_rows = db.execute(
        select(PastTopic.id, PastTopic.title, Teacher.id, Teacher.full_name, Teacher.position)
        .join(Teacher, Teacher.id == PastTopic.teacher_id)
    ).all()

    historical_past = [
        (row[1], row[2], teacher_table_name(row[3], row[4]), row[0])
        for row in past_rows
    ]
    current_meta = [
        (t.title, t.teacher_id, teacher_table_name(t.teacher.full_name, t.teacher.position), t.id)
        for t in current
    ]

    teacher_requests: list[tuple[str, list[str]]] = []
    global_requests: list[tuple[str, list[str]]] = []
    global_metadata: list[dict[str, str]] = []
    for topic in current:
        source_id = topic.source_past_topic_id

        # Колонка «Прошлые темы» сравнивает ТОЛЬКО с историческими PastTopic
        # этого преподавателя. Другие темы текущего набора сюда не попадают.
        teacher_candidates = [
            title for title, tid, _, past_id in historical_past
            if tid == topic.teacher_id and past_id != source_id
        ]

        # «По всей базе» = вся историческая база PastTopic + другие темы
        # текущего открытого набора. Старые сгенерированные наборы не накапливаем.
        global_entries = [
            (title, display) for title, _, display, past_id in historical_past
            if past_id != source_id
        ]
        global_entries += [
            (title, display) for title, _, display, oid in current_meta
            if oid != topic.id
        ]

        meta: dict[str, str] = {}
        for title, display in global_entries:
            meta.setdefault(title, display)
        teacher_requests.append((topic.title, list(dict.fromkeys(teacher_candidates))))
        global_requests.append((topic.title, list(meta.keys())))
        global_metadata.append(meta)

    runtime = load_runtime_settings(db)
    similarity = SimilarityService(runtime)
    results = await similarity.closest_bulk(teacher_requests + global_requests)
    teacher_results = results[:len(current)]
    global_results = results[len(current):]

    for index, topic in enumerate(current):
        t_score, t_closest, t_method = teacher_results[index]
        g_score, g_closest, g_method = global_results[index]
        topic.teacher_similarity_score = t_score
        topic.teacher_closest_topic = t_closest
        topic.teacher_similarity_method = t_method
        topic.global_similarity_score = g_score
        topic.global_closest_topic = g_closest
        topic.global_closest_teacher = global_metadata[index].get(g_closest or "") if g_closest else None
        topic.global_similarity_method = g_method
        topic.similarity_score = g_score
        topic.closest_past_topic = g_closest
        topic.similarity_method = g_method if g_method != "none" else t_method
    db.commit()


def _backfill_ready_topic_sources(db: Session) -> set[int]:
    """Привязывает готовые темы из v14 к их PastTopic без удаления данных.

    Возвращает ID наборов, у которых после привязки нужно один раз пересчитать
    похожесть. Привязываем только темы с явным маркером готовой темы, чтобы
    случайное текстовое совпадение обычной генерации не считалось self-source.
    """
    rows = db.scalars(
        select(GeneratedTopic).where(GeneratedTopic.source_past_topic_id.is_(None))
    ).all()
    changed_topic_ids: list[int] = []
    for topic in rows:
        rationale = (topic.rationale or "").casefold()
        keywords = {str(x).casefold() for x in (topic.keywords or [])}
        if "готовая тема из списка ранее одобренных" not in rationale and "готовая тема" not in keywords:
            continue
        normalized = re.sub(r"\s+", " ", topic.title.casefold()).strip().rstrip(".")
        candidates = db.scalars(
            select(PastTopic).where(PastTopic.teacher_id == topic.teacher_id)
        ).all()
        exact = [
            item for item in candidates
            if re.sub(r"\s+", " ", item.title.casefold()).strip().rstrip(".") == normalized
        ]
        if exact:
            # Если одна и та же тема исторически записана несколько раз, связываем
            # только с одной записью. Остальные останутся в сравнении как реальные дубликаты.
            topic.source_past_topic_id = exact[0].id
            changed_topic_ids.append(topic.id)

    if not changed_topic_ids:
        return set()
    db.commit()
    return set(db.scalars(
        select(GenerationBatchTopic.batch_id).where(GenerationBatchTopic.topic_id.in_(changed_topic_ids))
    ).all())


@app.on_event("startup")
async def repair_v14_ready_topic_self_matches() -> None:
    # Одноразовая мягкая миграция: уже созданные в v14 готовые темы тоже перестанут
    # сравниваться сами с собой после обновления, без удаления текущего набора.
    with Session(engine) as db:
        batch_ids = _backfill_ready_topic_sources(db)
        for batch_id in sorted(batch_ids):
            await _recalculate_batch_similarities(batch_id, db)


async def _generate_for_teacher(
    teacher: Teacher,
    count: int,
    db: Session,
    focus: str | None = None,
    extra_avoid: list[str] | None = None,
) -> tuple[list[GeneratedTopic], str, str | None]:
    approved_titles = [t.title for t in teacher.topics if t.status == "approved"]
    avoid_titles = list(dict.fromkeys([*approved_titles, *(extra_avoid or [])]))
    runtime = load_runtime_settings(db)
    style_examples = db.scalars(
        select(PastTopic.title).where(PastTopic.is_reference.is_(True)).order_by(PastTopic.id.desc()).limit(16)
    ).all()
    try:
        candidates, mode, warning = await TopicGenerator(runtime).generate(
            teacher, count, avoid_titles, focus, style_examples=style_examples
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    created: list[GeneratedTopic] = []
    for i, candidate in enumerate(candidates):
        topic = GeneratedTopic(
            teacher_id=teacher.id,
            title=candidate.title,
            rationale=candidate.rationale,
            keywords=candidate.keywords,
            source_past_topic_id=candidate.source_past_topic_id,
            generation_index=i,
        )
        db.add(topic)
        db.flush()
        created.append(topic)
    db.commit()
    for item in created:
        db.refresh(item)
        item.teacher = teacher
    return created, mode, warning


def _generation_groups(
    selections: list[GenerationSelection],
    max_size: int = 5,
    max_topics: int = 40,
) -> list[list[GenerationSelection]]:
    """Формирует устойчивые Mistral-пакеты по числу преподавателей и тем.

    Не больше 5 преподавателей и примерно не больше 40 тем в одном ответе.
    Например, 25 преподавателей × 10 тем будут разбиты примерно на 7 запросов,
    а 20 преподавателей × 5 тем — на 4 запроса.
    """
    groups: list[list[GenerationSelection]] = []
    current: list[GenerationSelection] = []
    current_topics = 0
    for item in selections:
        count = max(1, int(item.count))
        would_overflow = current and (len(current) >= max_size or current_topics + count > max_topics)
        if would_overflow:
            groups.append(current)
            current = []
            current_topics = 0
        current.append(item)
        current_topics += count
        if len(current) >= max_size or current_topics >= max_topics:
            groups.append(current)
            current = []
            current_topics = 0
    if current:
        groups.append(current)
    return groups


@app.post("/api/generate/selected", response_model=GenerationResult)
async def generate_selected(payload: GenerateSelectedRequest, db: Session = Depends(get_db)):
    ids = [item.teacher_id for item in payload.selections]
    teachers = db.scalars(teacher_query().where(Teacher.id.in_(ids))).unique().all()
    by_id = {teacher.id: teacher for teacher in teachers}
    missing = [teacher_id for teacher_id in ids if teacher_id not in by_id]
    if missing:
        raise HTTPException(404, f"Не найдены преподаватели: {missing}")

    batch = GenerationBatch(
        focus=(payload.focus or "").strip() or None,
        selections=[item.model_dump() for item in payload.selections],
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)

    all_topics: list[GeneratedTopic] = []
    modes: set[str] = set()
    warnings: list[str] = []
    try:
        batch_reserved_titles: list[str] = []
        runtime = load_runtime_settings(db)
        generator = TopicGenerator(runtime)
        style_examples = db.scalars(
            select(PastTopic.title).where(PastTopic.is_reference.is_(True)).order_by(PastTopic.id.desc()).limit(16)
        ).all()

        # v23: Mistral получает компактные пакеты до 5 преподавателей и примерно до 40 тем.
        # Это устойчивее для strict JSON и качества профилей при 5–10 темах на преподавателя.
        for group in _generation_groups(payload.selections, max_size=5, max_topics=40):
            teacher_requests = [(by_id[item.teacher_id], item.count) for item in group]
            try:
                generated_by_teacher, mode, group_warnings = await generator.generate_batch(
                    teacher_requests,
                    extra_avoid=batch_reserved_titles,
                    focus=batch.focus,
                    style_examples=style_examples,
                )
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc

            modes.add(mode)
            warnings.extend(group_warnings)
            for selection in group:
                teacher = by_id[selection.teacher_id]
                candidates = generated_by_teacher.get(teacher.id, [])
                for i, candidate in enumerate(candidates):
                    topic = GeneratedTopic(
                        teacher_id=teacher.id,
                        title=candidate.title,
                        rationale=candidate.rationale,
                        keywords=candidate.keywords,
                        source_past_topic_id=candidate.source_past_topic_id,
                        generation_index=i,
                    )
                    db.add(topic)
                    db.flush()
                    db.add(GenerationBatchTopic(batch_id=batch.id, topic_id=topic.id))
                    all_topics.append(topic)
                    batch_reserved_titles.append(topic.title)
            db.commit()

        await _recalculate_batch_similarities(batch.id, db)
        rank, name, generation_index, topic_id = topic_ordering()
        all_topics = db.scalars(
            topic_query()
            .join(Teacher, Teacher.id == GeneratedTopic.teacher_id)
            .join(GenerationBatchTopic, GenerationBatchTopic.topic_id == GeneratedTopic.id)
            .where(GenerationBatchTopic.batch_id == batch.id)
            .order_by(rank, name, generation_index, topic_id)
        ).unique().all()
    except Exception:
        db.rollback()
        linked_ids = db.scalars(select(GenerationBatchTopic.topic_id).where(GenerationBatchTopic.batch_id == batch.id)).all()
        db.query(GenerationBatchTopic).filter(GenerationBatchTopic.batch_id == batch.id).delete()
        if linked_ids:
            db.query(GeneratedTopic).filter(GeneratedTopic.id.in_(linked_ids)).delete(synchronize_session=False)
        db.delete(batch)
        db.commit()
        raise

    return GenerationResult(
        created=len(all_topics),
        mode="+".join(sorted(modes)),
        topics=[topic_to_schema(t) for t in all_topics],
        warning="\n".join(warnings) or None,
        batch_id=batch.id,
    )


# Старые endpoints оставлены для совместимости.
@app.post("/api/generate/all", response_model=GenerationResult)
async def generate_all(payload: GenerateRequest, db: Session = Depends(get_db)):
    rank, name = teacher_ordering()
    teachers = db.scalars(teacher_query().order_by(rank, name)).unique().all()
    if not teachers:
        raise HTTPException(400, "Сначала добавьте хотя бы одного преподавателя")
    selected = GenerateSelectedRequest(
        selections=[GenerationSelection(teacher_id=t.id, count=t.topic_count or 5) for t in teachers]
    )
    return await generate_selected(selected, db)


@app.post("/api/generate/teacher/{teacher_id}", response_model=GenerationResult)
async def generate_teacher(teacher_id: int, payload: GenerateRequest, db: Session = Depends(get_db)):
    teacher = db.get(Teacher, teacher_id)
    if not teacher:
        raise HTTPException(404, "Преподаватель не найден")
    selected = GenerateSelectedRequest(selections=[GenerationSelection(teacher_id=teacher_id, count=teacher.topic_count or 5)])
    return await generate_selected(selected, db)


def _batch_id_for_topic(db: Session, topic_id: int) -> int | None:
    return db.scalar(select(GenerationBatchTopic.batch_id).where(GenerationBatchTopic.topic_id == topic_id))


@app.post("/api/topics/{topic_id}/regenerate", response_model=TopicRead)
async def regenerate_topic(topic_id: int, db: Session = Depends(get_db)):
    topic = db.scalars(topic_query().where(GeneratedTopic.id == topic_id)).unique().one_or_none()
    if not topic:
        raise HTTPException(404, "Тема не найдена")
    teacher = db.scalars(teacher_query().where(Teacher.id == topic.teacher_id)).unique().one()
    batch_id = _batch_id_for_topic(db, topic.id)

    # Перегенерация обязана менять формулировку. Раньше текущая тема не попадала
    # в список запретов, поэтому demo-генератор (и иногда LLM) мог вернуть ровно
    # то же название, из-за чего кнопка визуально казалась нерабочей.
    avoid = [topic.title]
    avoid.extend(t.title for t in teacher.topics if t.id != topic.id)
    if batch_id:
        batch_titles = db.scalars(
            select(GeneratedTopic.title)
            .join(GenerationBatchTopic, GenerationBatchTopic.topic_id == GeneratedTopic.id)
            .where(GenerationBatchTopic.batch_id == batch_id, GeneratedTopic.id != topic.id)
        ).all()
        avoid.extend(batch_titles)
    avoid = list(dict.fromkeys(x.strip() for x in avoid if x and x.strip()))
    batch_focus = db.get(GenerationBatch, batch_id).focus if batch_id and db.get(GenerationBatch, batch_id) else None
    runtime = load_runtime_settings(db)
    style_examples = db.scalars(
        select(PastTopic.title).where(PastTopic.is_reference.is_(True)).order_by(PastTopic.id.desc()).limit(16)
    ).all()
    try:
        candidates, _, _ = await TopicGenerator(runtime).generate(teacher, 1, avoid, batch_focus, style_examples=style_examples)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    if not candidates:
        raise HTTPException(502, "Модель не вернула новую тему")
    candidate = candidates[0]
    topic.title = candidate.title
    topic.rationale = candidate.rationale
    topic.keywords = candidate.keywords
    topic.source_past_topic_id = candidate.source_past_topic_id
    topic.manual_edit = False
    topic.status = "draft"
    db.commit()
    if batch_id:
        await _recalculate_batch_similarities(batch_id, db)
    topic = db.scalars(topic_query().where(GeneratedTopic.id == topic_id)).unique().one()
    return topic_to_schema(topic)


@app.patch("/api/topics/{topic_id}", response_model=TopicRead)
async def edit_topic(topic_id: int, payload: TopicUpdate, db: Session = Depends(get_db)):
    topic = db.scalars(topic_query().where(GeneratedTopic.id == topic_id)).unique().one_or_none()
    if not topic:
        raise HTTPException(404, "Тема не найдена")
    topic.title = payload.title.strip()
    # После ручного изменения это уже не дословно перенесённая историческая тема.
    topic.source_past_topic_id = None
    topic.manual_edit = True
    db.commit()
    batch_id = _batch_id_for_topic(db, topic.id)
    if batch_id:
        await _recalculate_batch_similarities(batch_id, db)
    topic = db.scalars(topic_query().where(GeneratedTopic.id == topic_id)).unique().one()
    return topic_to_schema(topic)


@app.patch("/api/topics/{topic_id}/status", response_model=TopicRead)
async def change_topic_status(topic_id: int, payload: TopicStatusUpdate, db: Session = Depends(get_db)):
    topic = db.scalars(topic_query().where(GeneratedTopic.id == topic_id)).unique().one_or_none()
    if not topic:
        raise HTTPException(404, "Тема не найдена")
    topic.status = payload.status
    db.commit()
    batch_id = _batch_id_for_topic(db, topic.id)
    if batch_id:
        await _recalculate_batch_similarities(batch_id, db)
    topic = db.scalars(topic_query().where(GeneratedTopic.id == topic_id)).unique().one()
    return topic_to_schema(topic)


@app.post("/api/topics/approve-all")
def approve_all(batch_id: int | None = None, db: Session = Depends(get_db)):
    query = select(GeneratedTopic).where(GeneratedTopic.status == "draft")
    if batch_id is not None:
        query = query.join(GenerationBatchTopic, GenerationBatchTopic.topic_id == GeneratedTopic.id).where(
            GenerationBatchTopic.batch_id == batch_id
        )
    topics = db.scalars(query).all()
    for topic in topics:
        topic.status = "approved"
    db.commit()
    return {"approved": len(topics)}


@app.delete("/api/topics/{topic_id}", status_code=204)
async def delete_topic(topic_id: int, db: Session = Depends(get_db)):
    topic = db.get(GeneratedTopic, topic_id)
    if not topic:
        raise HTTPException(404, "Тема не найдена")
    batch_id = _batch_id_for_topic(db, topic.id)
    db.delete(topic)
    db.commit()
    if batch_id:
        await _recalculate_batch_similarities(batch_id, db)
    return Response(status_code=204)


def approved_topics(db: Session, batch_id: int | None = None) -> list[GeneratedTopic]:
    rank, name, generation_index, topic_id = topic_ordering()
    query = topic_query().join(Teacher, Teacher.id == GeneratedTopic.teacher_id).where(GeneratedTopic.status == "approved")
    if batch_id is not None:
        query = query.join(GenerationBatchTopic, GenerationBatchTopic.topic_id == GeneratedTopic.id).where(
            GenerationBatchTopic.batch_id == batch_id
        )
    query = query.order_by(rank, name, generation_index, topic_id)
    return db.scalars(query).unique().all()


@app.get("/api/export/xlsx")
def export_xlsx(batch_id: int | None = None, db: Session = Depends(get_db)):
    topics = approved_topics(db, batch_id)
    if not topics:
        raise HTTPException(400, "Сначала утвердите хотя бы одну тему")
    content = build_topics_xlsx(topics)
    suffix = f"_batch_{batch_id}" if batch_id is not None else ""
    return StreamingResponse(
        iter([content]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="vkr_topics_09_03_01{suffix}.xlsx"'},
    )


@app.post("/api/export/google-sheet", response_model=GoogleSheetResult)
def export_google_sheet(batch_id: int | None = None, db: Session = Depends(get_db)):
    topics = approved_topics(db, batch_id)
    if not topics:
        raise HTTPException(400, "Сначала утвердите хотя бы одну тему")
    try:
        from .services.google_sheets import GoogleSheetsService
        runtime = load_runtime_settings(db)
        spreadsheet_id, url, rows = GoogleSheetsService(runtime).create_topics_sheet(topics)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Google Sheets API error: {type(exc).__name__}: {exc}") from exc
    return GoogleSheetResult(spreadsheet_id=spreadsheet_id, spreadsheet_url=url, rows=rows)


# ------------------------- quality evaluation -------------------------

def evaluation_to_schema(run: QualityEvaluationRun) -> QualityEvaluationRead:
    return QualityEvaluationRead(
        id=run.id,
        batch_id=run.batch_id,
        reference_count=run.reference_count,
        topic_count=run.topic_count,
        mode=run.mode,
        metrics=run.metrics or {},
        items=[QualityEvaluationItem(**item) for item in (run.items or [])],
        created_at=run.created_at,
    )


@app.post("/api/evaluation/run", response_model=QualityEvaluationRead)
async def run_evaluation(batch_id: int, reference_count: int = 80, db: Session = Depends(get_db)):
    if reference_count < 50 or reference_count > 100:
        raise HTTPException(400, "Для эксперимента используйте от 50 до 100 эталонных тем")
    if not db.get(GenerationBatch, batch_id):
        raise HTTPException(404, "Набор генерации не найден")
    try:
        run = await run_quality_evaluation(db, batch_id, reference_count)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return evaluation_to_schema(run)


@app.get("/api/evaluation/latest", response_model=QualityEvaluationRead | None)
def latest_evaluation(batch_id: int | None = None, db: Session = Depends(get_db)):
    query = select(QualityEvaluationRun).order_by(QualityEvaluationRun.id.desc())
    if batch_id is not None:
        query = query.where(QualityEvaluationRun.batch_id == batch_id)
    run = db.scalars(query.limit(1)).one_or_none()
    return evaluation_to_schema(run) if run else None


@app.post("/api/evaluation/refresh-reference")
def refresh_reference(db: Session = Depends(get_db)):
    count = refresh_reference_set(db, 100)
    return {"reference_count": count}


# ------------------------- threshold calibration -------------------------

@app.post("/api/calibration/generate", response_model=list[CalibrationPairRead])
async def create_calibration_pairs(count: int = 28, db: Session = Depends(get_db)):
    try:
        return await generate_calibration_pairs(db, max(12, min(40, count)))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/calibration/pairs", response_model=list[CalibrationPairRead])
def list_calibration_pairs(db: Session = Depends(get_db)):
    return db.scalars(select(SimilarityCalibrationPair).order_by(SimilarityCalibrationPair.id)).all()


@app.patch("/api/calibration/pairs/{pair_id}", response_model=CalibrationPairRead)
def label_calibration_pair(pair_id: int, payload: CalibrationLabelUpdate, db: Session = Depends(get_db)):
    row = db.get(SimilarityCalibrationPair, pair_id)
    if not row:
        raise HTTPException(404, "Пара не найдена")
    row.label = payload.label
    db.commit()
    db.refresh(row)
    return row


@app.get("/api/calibration/summary", response_model=CalibrationSummary)
def calibration_summary(db: Session = Depends(get_db)):
    runtime = load_runtime_settings(db)
    green, yellow, accuracy, labeled = recommend_thresholds(db)
    total = db.scalar(select(func.count()).select_from(SimilarityCalibrationPair)) or 0
    return CalibrationSummary(
        total_pairs=int(total),
        labeled_pairs=labeled,
        current_green_max=runtime.similarity_green_max,
        current_yellow_max=runtime.similarity_yellow_max,
        recommended_green_max=green,
        recommended_yellow_max=yellow,
        accuracy=accuracy,
        calibrated=runtime.similarity_calibrated,
    )


@app.post("/api/calibration/apply", response_model=CalibrationSummary)
def apply_calibration(db: Session = Depends(get_db)):
    try:
        green, yellow, accuracy, labeled = apply_recommended_thresholds(db)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    total = db.scalar(select(func.count()).select_from(SimilarityCalibrationPair)) or 0
    return CalibrationSummary(
        total_pairs=int(total),
        labeled_pairs=labeled,
        current_green_max=green,
        current_yellow_max=yellow,
        recommended_green_max=green,
        recommended_yellow_max=yellow,
        accuracy=accuracy,
        calibrated=True,
    )
