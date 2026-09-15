"use client";

import { FormEvent, useEffect, useState } from "react";
import {
  type AppSettings,
  type AppSettingsUpdate,
  type IntegrationCheck,
  type IntegrationsStatus,
} from "../lib/api";

const FALLBACK_CHECK: IntegrationCheck = { status: "unconfigured", message: "не настроен", detail: null };
type AiCheckPayload = { api_key?: string | null; chat_model?: string | null; embedding_model?: string | null };

function StatusPill({ value }: { value: IntegrationCheck }) {
  const label = value.status === "connected" ? "Подключён" : value.status === "error" ? value.message : "Не настроен";
  return <span className={`connection-pill ${value.status}`}>{label}</span>;
}

export default function SettingsModal({
  settings,
  integrations,
  onClose,
  onSave,
  onCheckMistral,
  onCheckGoogle,
}: {
  settings: AppSettings;
  integrations?: IntegrationsStatus | null;
  onClose: () => void;
  onSave: (payload: AppSettingsUpdate) => Promise<void>;
  onCheckMistral: (payload: AiCheckPayload) => Promise<IntegrationCheck>;
  onCheckGoogle: (payload: { service_account_json?: string | null; spreadsheet_id?: string | null }) => Promise<IntegrationCheck>;
}) {
  const [tab, setTab] = useState<"ai" | "google" | "generation">("ai");
  const [busy, setBusy] = useState(false);
  const [checking, setChecking] = useState<"mistral" | "google" | null>(null);
  const [clearMistral, setClearMistral] = useState(false);
  const [clearGoogle, setClearGoogle] = useState(false);
  const [mistralCheck, setMistralCheck] = useState<IntegrationCheck>(integrations?.mistral || FALLBACK_CHECK);
  const [googleCheck, setGoogleCheck] = useState<IntegrationCheck>(integrations?.google_sheets || FALLBACK_CHECK);
  const [form, setForm] = useState({
    mistral_api_key: "",
    mistral_chat_model: settings.mistral_chat_model,
    mistral_embedding_model: settings.mistral_embedding_model,
    google_service_account_json: "",
    google_share_with_email: settings.google_share_with_email || "",
    google_spreadsheet_id: settings.google_spreadsheet_id || "",
    default_topic_count: settings.default_topic_count || 5,
    default_generation_focus: settings.default_generation_focus || "",
  });

  useEffect(() => {
    setForm((current) => ({
      ...current,
      mistral_chat_model: settings.mistral_chat_model,
      mistral_embedding_model: settings.mistral_embedding_model,
      google_share_with_email: settings.google_share_with_email || "",
      google_spreadsheet_id: settings.google_spreadsheet_id || "",
      default_topic_count: settings.default_topic_count || 5,
      default_generation_focus: settings.default_generation_focus || "",
    }));
  }, [settings]);

  useEffect(() => {
    if (integrations?.mistral) setMistralCheck(integrations.mistral);
    if (integrations?.google_sheets) setGoogleCheck(integrations.google_sheets);
  }, [integrations]);

  const set = (key: string, value: string | number) => setForm((f) => ({ ...f, [key]: value }));

  async function checkMistral() {
    setChecking("mistral");
    try {
      const result = await onCheckMistral({
        api_key: form.mistral_api_key.trim() || null,
        chat_model: form.mistral_chat_model.trim() || null,
        embedding_model: form.mistral_embedding_model.trim() || null,
      });
      setMistralCheck(result);
    } finally { setChecking(null); }
  }

  async function checkGoogle() {
    setChecking("google");
    try {
      const result = await onCheckGoogle({
        service_account_json: form.google_service_account_json.trim() || null,
        spreadsheet_id: form.google_spreadsheet_id.trim() || null,
      });
      setGoogleCheck(result);
    } finally { setChecking(null); }
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    try {
      if (form.mistral_api_key.trim() && !clearMistral) {
        const checked = await onCheckMistral({
          api_key: form.mistral_api_key.trim(),
          chat_model: form.mistral_chat_model.trim() || null,
          embedding_model: form.mistral_embedding_model.trim() || null,
        });
        setMistralCheck(checked);
        if (checked.status !== "connected") { setTab("ai"); return; }
      }
      if (form.google_service_account_json.trim() && !clearGoogle) {
        const checked = await onCheckGoogle({
          service_account_json: form.google_service_account_json.trim(),
          spreadsheet_id: form.google_spreadsheet_id.trim() || null,
        });
        setGoogleCheck(checked);
        if (checked.status !== "connected") { setTab("google"); return; }
      }

      const payload: AppSettingsUpdate = {
        mistral_chat_model: form.mistral_chat_model.trim(),
        mistral_embedding_model: form.mistral_embedding_model.trim(),
        google_share_with_email: form.google_share_with_email.trim() || null,
        google_spreadsheet_id: form.google_spreadsheet_id.trim() || null,
        default_topic_count: Number(form.default_topic_count) || 5,
        default_generation_focus: form.default_generation_focus.trim() || null,
      };
      if (clearMistral) payload.mistral_api_key = null;
      else if (form.mistral_api_key.trim()) payload.mistral_api_key = form.mistral_api_key.trim();
      if (clearGoogle) payload.google_service_account_json = null;
      else if (form.google_service_account_json.trim()) payload.google_service_account_json = form.google_service_account_json.trim();
      await onSave(payload);
      onClose();
    } finally { setBusy(false); }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal settings-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div><div className="eyebrow">Параметры проекта</div><h2>Настройки</h2></div>
          <button className="icon-btn" onClick={onClose} aria-label="Закрыть">×</button>
        </div>

        <div className="modal-tabs settings-tabs">
          <button type="button" className={tab === "ai" ? "modal-tab active" : "modal-tab"} onClick={() => setTab("ai")}>ИИ</button>
          <button type="button" className={tab === "google" ? "modal-tab active" : "modal-tab"} onClick={() => setTab("google")}>Google Sheets</button>
          <button type="button" className={tab === "generation" ? "modal-tab active" : "modal-tab"} onClick={() => setTab("generation")}>Генерация</button>
        </div>

        <form onSubmit={submit}>
          {tab === "ai" && (
            <div className="settings-section">
              <div className="settings-status-row primary-provider">
                <div><b>Mistral API</b><span>Генерация новых тем + дополнительный семантический сигнал для проверки сходства</span></div>
                <StatusPill value={mistralCheck}/>
              </div>
              {mistralCheck.detail && <div className={`connection-detail ${mistralCheck.status}`}>{mistralCheck.detail}</div>}
              <label className="field wide">
                <span>Mistral API Key</span>
                <input type="password" autoComplete="off" value={form.mistral_api_key} onChange={(e) => { set("mistral_api_key", e.target.value); setClearMistral(false); }} placeholder={settings.mistral_configured && !clearMistral ? `Ключ уже сохранён ${settings.mistral_key_hint || ""} — оставьте пустым, чтобы не менять` : "Вставьте API key из Mistral Studio"} />
              </label>
              {settings.mistral_configured && !clearMistral && <button type="button" className="credential-reset" onClick={() => { setClearMistral(true); set("mistral_api_key", ""); setMistralCheck(FALLBACK_CHECK); }}>Сбросить сохранённый ключ Mistral</button>}
              {clearMistral && <div className="credential-reset-note">Ключ Mistral будет удалён после сохранения настроек.</div>}
              <div className="grid-2">
                <label className="field"><span>Модель генерации</span><input value={form.mistral_chat_model} onChange={(e) => set("mistral_chat_model", e.target.value)} placeholder="mistral-small-latest" /></label>
                <label className="field"><span>Модель эмбеддингов</span><input value={form.mistral_embedding_model} onChange={(e) => set("mistral_embedding_model", e.target.value)} placeholder="mistral-embed" /></label>
              </div>
              <button type="button" className="btn secondary connection-test-btn" onClick={checkMistral} disabled={checking === "mistral"}>{checking === "mistral" ? "Проверяем…" : "Проверить Mistral"}</button>
              <div className="ai-routing-card">
                <div><span>Основная модель</span><b>{form.mistral_chat_model || "mistral-small-latest"}</b></div>
                <div className="routing-arrow">→</div>
                <div><span>Embeddings</span><b>{form.mistral_embedding_model || "mistral-embed"}</b></div>
              </div>
              <div className="settings-note">Mistral работает в общей последовательной очереди: не чаще одного запроса каждые 1,25 секунды. При временном HTTP 429 backend автоматически повторяет запрос через 2, 4 и 8 секунд. Генерация тем и проверка сходства выполняются пакетно; если embeddings недоступны, similarity переходит на локальный режим.</div>
            </div>
          )}

          {tab === "google" && (
            <div className="settings-section">
              <div className="settings-status-row">
                <div><b>Google Sheets</b><span>Публикация утверждённых тем</span></div>
                <StatusPill value={googleCheck}/>
              </div>
              {googleCheck.detail && <div className={`connection-detail ${googleCheck.status}`}>{googleCheck.detail}</div>}
              <label className="field wide">
                <span>JSON сервисного аккаунта Google</span>
                <textarea rows={7} value={form.google_service_account_json} onChange={(e) => { set("google_service_account_json", e.target.value); setClearGoogle(false); }} placeholder={settings.google_service_account_configured && !clearGoogle ? "Данные уже сохранены — оставьте поле пустым, чтобы не менять" : "Вставьте содержимое JSON сервисного аккаунта"} />
              </label>
              {settings.google_service_account_configured && !clearGoogle && <button type="button" className="credential-reset" onClick={() => { setClearGoogle(true); set("google_service_account_json", ""); setGoogleCheck(FALLBACK_CHECK); }}>Сбросить данные Google</button>}
              {clearGoogle && <div className="credential-reset-note">Данные сервисного аккаунта будут удалены после сохранения настроек.</div>}
              <div className="grid-2">
                <label className="field"><span>ID Google-таблицы <i>необязательно</i></span><input value={form.google_spreadsheet_id} onChange={(e) => set("google_spreadsheet_id", e.target.value)} placeholder="1AbC...xyz" /></label>
                <label className="field"><span>Email для доступа <i>необязательно</i></span><input type="email" value={form.google_share_with_email} onChange={(e) => set("google_share_with_email", e.target.value)} placeholder="admin@example.com" /></label>
              </div>
              <button type="button" className="btn secondary connection-test-btn" onClick={checkGoogle} disabled={checking === "google"}>{checking === "google" ? "Проверяем…" : "Проверить подключение"}</button>
              <div className="settings-note">Если ID таблицы указан, проверяется реальный доступ на редактирование. Без ID проверяется авторизация сервисного аккаунта и связь с Google API.</div>
            </div>
          )}

          {tab === "generation" && (
            <div className="settings-section">
              <div className="settings-program"><span>Направление</span><b>09.03.01 «Информатика и вычислительная техника»</b></div>
              <div className="grid-2">
                <label className="field"><span>Тем по умолчанию на преподавателя</span><input type="number" min={1} max={50} value={form.default_topic_count} onChange={(e) => set("default_topic_count", Number(e.target.value))} /></label>
              </div>
              <label className="field wide"><span>Фокус генерации по умолчанию <i>необязательно</i></span><textarea rows={5} value={form.default_generation_focus} onChange={(e) => set("default_generation_focus", e.target.value)} placeholder="Например: компьютерное зрение; веб-системы; открытые датасеты…" /></label>
              <div className="settings-note similarity-note"><b>Проверка сходства теперь автоматическая.</b> Система сочетает содержательные слова темы и Mistral Embeddings как дополнительный смысловой сигнал и отдельно отмечает точные дубликаты. Ручная калибровка больше не нужна.</div>
            </div>
          )}

          <div className="modal-actions settings-actions">
            <button type="button" className="btn ghost" onClick={onClose}>Отмена</button>
            <button type="submit" className="btn primary" disabled={busy}>{busy ? "Сохраняем…" : "Сохранить настройки"}</button>
          </div>
        </form>
      </div>
    </div>
  );
}
