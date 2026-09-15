"use client";

import { FormEvent, KeyboardEvent, useEffect, useState } from "react";
import { api, type Teacher } from "../lib/api";

export type TeacherFormPayload = {
  full_name: string;
  topic_count: number;
  department: string | null;
  position: string | null;
  email: string | null;
  telegram: string | null;
  phone: string | null;
  research_areas: string[];
  orcid: string | null;
  scopus_id: string | null;
  past_topics?: { title: string; year?: number }[];
};

const DEPARTMENTS = ["САПРиПК", "ЭВМиС"];
const POSITIONS = ["преподаватель", "доцент", "профессор"];
const FALLBACK_RESEARCH_AREAS = [
  "Системы искусственного интеллекта",
  "Машинное обучение и анализ данных",
  "Системный анализ",
  "Компьютерное зрение",
  "Видеоаналитика",
  "Робототехника",
  "Интернет вещей и встраиваемые системы",
  "Мультимедийные и игровые технологии",
  "Разработка обучающих игр",
  "Ассистивные технологии",
  "Чат-боты и диалоговые системы",
  "Компьютерная лингвистика и NLP",
  "Речевые и аудиотехнологии",
  "Информационные системы и веб-технологии",
  "Кибербезопасность",
  "Компьютерные сети и распределённые системы",
  "Геоинформационные системы",
  "Алгоритмы и оптимизация",
  "Мобильные технологии",
  "Базы данных и управление данными",
];

const text = (value?: string | null) => value || "";

function departmentList(value: string) {
  return value.split(/[,;/+|]/).map((x) => x.trim()).filter(Boolean);
}

export default function TeacherModal({
  teacher,
  onClose,
  onSave,
}: {
  teacher?: Teacher | null;
  onClose: () => void;
  onSave: (payload: TeacherFormPayload) => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [researchInput, setResearchInput] = useState("");
  const [researchSuggestions, setResearchSuggestions] = useState<string[]>(FALLBACK_RESEARCH_AREAS);
  const [form, setForm] = useState({
    full_name: "",
    department: "",
    position: "",
    email: "",
    telegram: "",
    phone: "",
    research_areas: [] as string[],
    orcid: "",
    scopus_id: "",
    past_topics: "",
  });

  useEffect(() => {
    api.researchAreas()
      .then((data) => {
        if (data.items?.length) setResearchSuggestions(data.items);
      })
      .catch(() => {
        // Если backend временно недоступен, встроенные подсказки всё равно остаются.
      });
  }, []);

  useEffect(() => {
    setResearchInput("");
    if (!teacher) {
      setForm({
        full_name: "",
        department: "",
        position: "",
        email: "",
        telegram: "",
        phone: "",
        research_areas: [],
        orcid: "",
        scopus_id: "",
        past_topics: "",
      });
      return;
    }
    setForm({
      full_name: teacher.full_name,
      department: text(teacher.department),
      position: text(teacher.position),
      email: text(teacher.email),
      telegram: text(teacher.telegram),
      phone: text(teacher.phone),
      research_areas: teacher.research_areas || [],
      orcid: text(teacher.orcid),
      scopus_id: text(teacher.scopus_id),
      past_topics: teacher.past_topics.map((x) => x.title).join("\n"),
    });
  }, [teacher]);

  const set = (key: string, value: string) => setForm((f) => ({ ...f, [key]: value }));

  function toggleDepartment(department: string) {
    setForm((current) => {
      const selected = departmentList(current.department);
      const next = selected.includes(department)
        ? selected.filter((item) => item !== department)
        : [...selected, department];
      return { ...current, department: DEPARTMENTS.filter((item) => next.includes(item)).join(", ") };
    });
  }

  function normalizedArea(value: string) {
    return value.trim().replace(/\s+/g, " ");
  }

  function appendResearchArea(raw = researchInput) {
    const area = normalizedArea(raw);
    if (!area) return;
    setForm((current) => {
      const exists = current.research_areas.some((item) => item.toLocaleLowerCase("ru") === area.toLocaleLowerCase("ru"));
      if (exists) return current;
      return { ...current, research_areas: [...current.research_areas, area] };
    });
    setResearchInput("");
  }

  function removeResearchArea(area: string) {
    setForm((current) => ({ ...current, research_areas: current.research_areas.filter((item) => item !== area) }));
  }

  function researchKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      appendResearchArea();
    }
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    const normalizedName = form.full_name.trim().replace(/\s+/g, " ");
    if (normalizedName.split(" ").length < 3) {
      setFormError("Укажите ФИО полностью: Фамилия Имя Отчество");
      return;
    }

    const pendingArea = normalizedArea(researchInput);
    const researchAreas = [...form.research_areas];
    if (pendingArea && !researchAreas.some((item) => item.toLocaleLowerCase("ru") === pendingArea.toLocaleLowerCase("ru"))) {
      researchAreas.push(pendingArea);
    }

    setFormError(null);
    setBusy(true);
    try {
      await onSave({
        full_name: normalizedName,
        topic_count: teacher?.topic_count || 5,
        department: form.department || null,
        position: form.position || null,
        email: form.email.trim() || null,
        telegram: form.telegram.trim() || null,
        phone: form.phone.trim() || null,
        research_areas: researchAreas,
        orcid: form.orcid.trim() || null,
        scopus_id: form.scopus_id.trim() || null,
        past_topics: form.past_topics
          .split("\n")
          .map((x) => x.trim())
          .filter(Boolean)
          .map((title) => ({ title })),
      });
      onClose();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : "Не удалось сохранить преподавателя");
    } finally {
      setBusy(false);
    }
  }

  const selectedDepartments = departmentList(form.department);
  const legacyDepartment = selectedDepartments.some((item) => !DEPARTMENTS.includes(item));
  const legacyPosition = form.position && !POSITIONS.includes(form.position.toLowerCase());
  const researchQuery = normalizedArea(researchInput).toLocaleLowerCase("ru");
  const visibleSuggestions = researchSuggestions.filter((item) =>
    !researchQuery || item.toLocaleLowerCase("ru").includes(researchQuery),
  );

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal teacher-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="eyebrow">База преподавателей</div>
            <h2>{teacher ? "Редактирование преподавателя" : "Новый преподаватель"}</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Закрыть">×</button>
        </div>

        <form onSubmit={submit}>
          <div className="teacher-form-section">
            <div className="section-title">Основные данные</div>
            <div className="grid-2">
              <label className="field wide">
                <span>ФИО преподавателя <b>*</b></span>
                <input
                  value={form.full_name}
                  onChange={(e) => set("full_name", e.target.value)}
                  required
                  placeholder="Например: Скоробогатченко Дмитрий Анатольевич"
                  autoFocus
                />
              </label>
              {formError && <div className="form-error field wide">{formError}</div>}
              <div className="field department-field">
                <span>Кафедра</span>
                <div className="department-options">
                  {DEPARTMENTS.map((item) => {
                    const checked = selectedDepartments.includes(item);
                    return (
                      <label key={item} className={`department-option ${checked ? "selected" : ""}`}>
                        <input type="checkbox" checked={checked} onChange={() => toggleDepartment(item)} />
                        <span className="department-check">{checked ? "✓" : ""}</span>
                        <b>{item}</b>
                      </label>
                    );
                  })}
                </div>
                <small className="department-help">Можно выбрать одну кафедру или обе.</small>
                {legacyDepartment && <small className="department-help warning">Старое значение кафедры будет заменено после сохранения.</small>}
              </div>
              <label className="field">
                <span>Должность</span>
                <select value={form.position} onChange={(e) => set("position", e.target.value)}>
                  <option value="">Не указана</option>
                  {legacyPosition && <option value={form.position}>{form.position} (старое значение)</option>}
                  {POSITIONS.map((item) => <option key={item} value={item}>{item}</option>)}
                </select>
              </label>
            </div>
          </div>

          <div className="teacher-form-section">
            <div className="section-title">Контакты</div>
            <div className="grid-2">
              <label className="field"><span>Email</span><input type="email" value={form.email} onChange={(e) => set("email", e.target.value)} /></label>
              <label className="field"><span>Telegram</span><input value={form.telegram} onChange={(e) => set("telegram", e.target.value)} placeholder="@username" /></label>
              <label className="field"><span>Телефон</span><input value={form.phone} onChange={(e) => set("phone", e.target.value)} /></label>
            </div>
          </div>

          <div className="teacher-form-section scientific-profile-section">
            <div className="section-title-row scientific-profile-title">
              <div>
                <div className="section-title">Научный профиль <span className="optional-label">необязательно</span></div>
                <p>Выберите направления, которые лучше всего описывают профиль преподавателя. Они помогают Mistral учитывать профиль преподавателя при генерации тем ВКР.</p>
              </div>
              <span className="research-selected-count">{form.research_areas.length} выбрано</span>
            </div>

            <div className="research-profile-picker">
              <div className="research-profile-search">
                <span className="research-search-icon">⌕</span>
                <input
                  value={researchInput}
                  onChange={(e) => setResearchInput(e.target.value)}
                  onKeyDown={researchKeyDown}
                  placeholder="Поиск направления или своя формулировка…"
                  autoComplete="off"
                />
                <button type="button" className="research-custom-add" onClick={() => appendResearchArea()} disabled={!normalizedArea(researchInput)}>+ Добавить</button>
              </div>

              {form.research_areas.length > 0 && (
                <div className="research-selected-zone">
                  <span className="research-zone-label">Выбрано</span>
                  <div className="research-area-chips">
                    {form.research_areas.map((area) => (
                      <span className="research-area-chip" key={area}>
                        <span>{area}</span>
                        <button type="button" onClick={() => removeResearchArea(area)} aria-label={`Удалить ${area}`}>×</button>
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="research-catalog-head">
                <span>Направления</span>
                <small>{researchQuery ? `Найдено: ${visibleSuggestions.length}` : "20 базовых направлений"}</small>
              </div>
              <div className="research-area-grid">
                {visibleSuggestions.map((item) => {
                  const selected = form.research_areas.some((area) => area.toLocaleLowerCase("ru") === item.toLocaleLowerCase("ru"));
                  return (
                    <button
                      type="button"
                      key={item}
                      className={`research-area-option ${selected ? "selected" : ""}`}
                      onClick={() => selected ? removeResearchArea(form.research_areas.find((area) => area.toLocaleLowerCase("ru") === item.toLocaleLowerCase("ru")) || item) : appendResearchArea(item)}
                    >
                      <span className="research-option-check">{selected ? "✓" : "+"}</span>
                      <span>{item}</span>
                    </button>
                  );
                })}
                {!visibleSuggestions.length && (
                  <div className="research-empty-result">Нет готового направления. Нажмите «+ Добавить», чтобы сохранить свою формулировку.</div>
                )}
              </div>
              <div className="research-picker-help">Можно выбрать несколько направлений. Свои формулировки тоже разрешены; Enter добавляет введённое значение.</div>
            </div>

            <div className="grid-2 research-identifiers">
              <label className="field">
                <span>ORCID</span>
                <input value={form.orcid} onChange={(e) => set("orcid", e.target.value)} placeholder="0000-0000-0000-0000" />
              </label>
              <label className="field">
                <span>Scopus Author ID</span>
                <input value={form.scopus_id} onChange={(e) => set("scopus_id", e.target.value)} placeholder="Например: 54978648200" />
              </label>
            </div>
          </div>

          <div className="teacher-form-section approved-history-section">
            <div className="section-title-row">
              <div>
                <div className="section-title">Одобренные темы ВКР прошлых лет</div>
              </div>
              <span className="history-count">{form.past_topics.split("\n").filter((x) => x.trim()).length} тем</span>
            </div>
            <label className="field wide">
              <textarea
                className="past-topics-editor"
                rows={10}
                value={form.past_topics}
                onChange={(e) => set("past_topics", e.target.value)}
                placeholder={"По одной теме на строку. Например:\nРазработка системы видеоаналитики спортивного матча с автоматическим формированием статистики\nРазработка информационной системы контроля выполнения выпускных квалификационных работ"}
              />
            </label>
          </div>

          <div className="modal-actions">
            <button type="button" className="btn ghost" onClick={onClose}>Отмена</button>
            <button type="submit" className="btn primary" disabled={busy}>{busy ? "Сохранение…" : "Сохранить в базе"}</button>
          </div>
        </form>
      </div>
    </div>
  );
}
