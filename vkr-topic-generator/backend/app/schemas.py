from datetime import datetime
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PastTopicCreate(BaseModel):
    title: str = Field(min_length=5, max_length=500)
    year: int | None = Field(default=None, ge=1990, le=2100)


class PastTopicRead(PastTopicCreate):
    id: int
    source_file: str | None = None
    source_sheet: str | None = None
    source_row: int | None = None
    source_project: str | None = None
    is_reference: bool = False
    model_config = ConfigDict(from_attributes=True)


class TeacherBase(BaseModel):
    full_name: str = Field(min_length=3, max_length=255)
    topic_count: int = Field(default=5, ge=1, le=50)
    department: str | None = None
    position: str | None = None
    email: str | None = None
    telegram: str | None = None
    phone: str | None = None
    research_areas: list[str] = []
    orcid: str | None = None
    scopus_id: str | None = None

    @field_validator("research_areas")
    @classmethod
    def clean_lists(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class TeacherWriteBase(TeacherBase):
    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if len(normalized.split()) < 3:
            raise ValueError("Укажите ФИО полностью: Фамилия Имя Отчество")
        return normalized

    @field_validator("department")
    @classmethod
    def validate_department(cls, value: str | None) -> str | None:
        if not value:
            return None
        raw = value.strip()
        # Для совместимости с существующей БД несколько кафедр хранятся в одном
        # поле, но UI работает как настоящий мультивыбор.
        parts = [item.strip() for item in re.split(r"[,;/+|]", raw) if item.strip()]
        allowed = ["САПРиПК", "ЭВМиС"]
        unique = []
        for item in parts:
            if item not in allowed:
                raise ValueError("Кафедра должна быть САПРиПК и/или ЭВМиС")
            if item not in unique:
                unique.append(item)
        return ", ".join(item for item in allowed if item in unique) or None

    @field_validator("position")
    @classmethod
    def validate_position(cls, value: str | None) -> str | None:
        if not value:
            return None
        value = value.strip().lower()
        if value not in {"преподаватель", "доцент", "профессор"}:
            raise ValueError("Выберите должность: преподаватель, доцент или профессор")
        return value


class TeacherCreate(TeacherWriteBase):
    past_topics: list[PastTopicCreate] = []


class TeacherUpdate(TeacherWriteBase):
    past_topics: list[PastTopicCreate] | None = None


class TeacherRead(TeacherBase):
    id: int
    past_topics: list[PastTopicRead] = []
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class PastTopicBackup(PastTopicCreate):
    source_file: str | None = None
    source_sheet: str | None = None
    source_row: int | None = None
    source_project: str | None = None
    is_reference: bool = False


class TeacherBackupEntry(TeacherWriteBase):
    past_topics: list[PastTopicBackup] = []


class TeacherImportPayload(BaseModel):
    format: Literal["vkr-ai-teachers"] = "vkr-ai-teachers"
    version: Literal[1, 2] = 2
    teachers: list[TeacherBackupEntry] = Field(min_length=1, max_length=1000)

    @field_validator("teachers")
    @classmethod
    def unique_names(cls, value: list[TeacherBackupEntry]) -> list[TeacherBackupEntry]:
        names = [" ".join(item.full_name.split()).casefold() for item in value]
        if len(names) != len(set(names)):
            raise ValueError("В JSON есть повторяющиеся преподаватели с одинаковым ФИО")
        return value


class TeacherImportResult(BaseModel):
    created: int
    updated: int
    total: int


class TopicRead(BaseModel):
    id: int
    teacher_id: int
    teacher_name: str
    teacher_contact: str
    title: str
    rationale: str | None
    keywords: list[str]

    # Совместимое поле = глобальная похожесть.
    similarity_score: float
    closest_past_topic: str | None
    teacher_similarity_score: float
    teacher_closest_topic: str | None
    global_similarity_score: float
    global_closest_topic: str | None
    global_closest_teacher: str | None
    similarity_method: str
    teacher_similarity_method: str = "none"
    global_similarity_method: str = "none"

    status: str
    manual_edit: bool
    created_at: datetime
    updated_at: datetime


class TopicUpdate(BaseModel):
    title: str = Field(min_length=8, max_length=500)


class TopicStatusUpdate(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in {"draft", "approved", "rejected"}:
            raise ValueError("status must be draft, approved or rejected")
        return value


class GenerateRequest(BaseModel):
    replace_drafts: bool = True


class GenerationSelection(BaseModel):
    teacher_id: int = Field(ge=1)
    count: int = Field(ge=1, le=50)


class GenerateSelectedRequest(BaseModel):
    selections: list[GenerationSelection] = Field(min_length=1, max_length=100)
    focus: str | None = Field(default=None, max_length=1500)

    @field_validator("selections")
    @classmethod
    def unique_teachers(cls, value: list[GenerationSelection]) -> list[GenerationSelection]:
        ids = [item.teacher_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Один преподаватель не должен встречаться в выборе дважды")
        return value


class GenerationResult(BaseModel):
    created: int
    mode: str
    topics: list[TopicRead]
    warning: str | None = None
    batch_id: int | None = None


class GenerationBatchRead(BaseModel):
    id: int
    focus: str | None
    selections: list[GenerationSelection]
    topic_count: int
    teacher_count: int
    approved_count: int
    attention_count: int
    created_at: datetime


class GoogleSheetResult(BaseModel):
    spreadsheet_id: str
    spreadsheet_url: str
    rows: int


class AppSettingsRead(BaseModel):
    mistral_configured: bool
    mistral_key_hint: str | None = None
    mistral_chat_model: str
    mistral_embedding_model: str
    google_service_account_configured: bool
    google_service_account_hint: str | None = None
    google_share_with_email: str | None = None
    google_spreadsheet_id: str | None = None
    default_topic_count: int = 5
    default_generation_focus: str | None = None
    similarity_green_max: float = 45.0
    similarity_yellow_max: float = 70.0
    similarity_calibrated: bool = True


class AppSettingsUpdate(BaseModel):
    mistral_api_key: str | None = Field(default=None, max_length=1000)
    mistral_chat_model: str | None = Field(default=None, max_length=255)
    mistral_embedding_model: str | None = Field(default=None, max_length=255)
    google_service_account_json: str | None = Field(default=None, max_length=20000)
    google_share_with_email: str | None = Field(default=None, max_length=255)
    google_spreadsheet_id: str | None = Field(default=None, max_length=255)
    default_topic_count: int | None = Field(default=None, ge=1, le=50)
    default_generation_focus: str | None = Field(default=None, max_length=1500)
    similarity_green_max: float | None = Field(default=None, ge=0, le=99)
    similarity_yellow_max: float | None = Field(default=None, ge=1, le=100)


class IntegrationCheckRead(BaseModel):
    status: Literal["unconfigured", "connected", "error"]
    message: str
    detail: str | None = None


class IntegrationsStatusRead(BaseModel):
    mistral: IntegrationCheckRead
    google_sheets: IntegrationCheckRead


class MistralCheckRequest(BaseModel):
    api_key: str | None = Field(default=None, max_length=1000)
    chat_model: str | None = Field(default=None, max_length=255)
    embedding_model: str | None = Field(default=None, max_length=255)


class GoogleCheckRequest(BaseModel):
    service_account_json: str | None = Field(default=None, max_length=20000)
    spreadsheet_id: str | None = Field(default=None, max_length=255)


class HistoryDeleteResult(BaseModel):
    deleted_batches: int
    deleted_topics: int


class HistoryImportSheet(BaseModel):
    name: str
    detected_rows: int
    matched_rows: int
    unknown_rows: int


class HistoryImportItem(BaseModel):
    sheet: str
    row: int
    teacher_name: str
    title: str
    year: int | None = None
    project: str | None = None
    matched_teacher_id: int | None = None
    matched_teacher_name: str | None = None
    match_type: str
    already_exists: bool = False
    recommended_reference: bool = False


class HistoryImportPreview(BaseModel):
    filename: str
    total_detected: int
    matched: int
    unknown: int
    duplicates: int
    recommended_reference: int
    sheets: list[HistoryImportSheet]
    items: list[HistoryImportItem]


class HistoryImportResult(BaseModel):
    imported: int
    skipped_duplicates: int
    skipped_unknown: int
    created_teachers: int
    reference_count: int


class QualityEvaluationItem(BaseModel):
    topic_id: int
    title: str
    teacher_name: str
    teacher_fit: float
    specificity: float
    practicality: float
    novelty: float
    overall: float


class QualityEvaluationRead(BaseModel):
    id: int
    batch_id: int
    reference_count: int
    topic_count: int
    mode: str
    metrics: dict
    items: list[QualityEvaluationItem]
    created_at: datetime


CalibrationLabel = Literal["different", "similar", "duplicate"]


class CalibrationPairRead(BaseModel):
    id: int
    left_title: str
    right_title: str
    similarity_score: float
    similarity_method: str
    label: CalibrationLabel | None = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class CalibrationLabelUpdate(BaseModel):
    label: CalibrationLabel | None


class CalibrationSummary(BaseModel):
    total_pairs: int
    labeled_pairs: int
    current_green_max: float
    current_yellow_max: float
    recommended_green_max: float | None = None
    recommended_yellow_max: float | None = None
    accuracy: float | None = None
    calibrated: bool = False
