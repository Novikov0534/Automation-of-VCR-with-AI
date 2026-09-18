"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import TeacherModal, { TeacherFormPayload } from "../components/TeacherModal";
import TopicEditModal from "../components/TopicEditModal";
import SettingsModal from "../components/SettingsModal";
import HistoryImportModal from "../components/HistoryImportModal";
import TeacherDeleteModal from "../components/TeacherDeleteModal";
import HoldToClearButton from "../components/HoldToClearButton";
import {
  API_URL, api, AppSettings, AppSettingsUpdate, GenerationBatch, GenerationProgressEvent, HistoryImportPreview,
  IntegrationCheck, IntegrationsStatus, PastTopic, Teacher, Topic,
} from "../lib/api";

type Tab = "teachers" | "generation" | "topics" | "history" | "publish";

const DEFAULT_APP_SETTINGS: AppSettings = {
  mistral_configured: false,
  mistral_key_hint: null,
  mistral_chat_model: "ministral-8b-2512",
  mistral_embedding_model: "mistral-embed",
  google_service_account_configured: false,
  google_service_account_hint: null,
  google_share_with_email: null,
  google_spreadsheet_id: null,
  default_topic_count: 5,
  default_generation_focus: null,
  similarity_green_max: 45,
  similarity_yellow_max: 70,
  similarity_calibrated: true,
};

const POSITION_RANK: Record<string, number> = { "профессор": 0, "доцент": 1, "преподаватель": 2 };
function sortTeachers(items: Teacher[]) {
  return [...items].sort((a, b) => {
    const rank = (POSITION_RANK[(a.position || "").toLowerCase()] ?? 3) - (POSITION_RANK[(b.position || "").toLowerCase()] ?? 3);
    return rank || a.full_name.localeCompare(b.full_name, "ru");
  });
}

function isExactDuplicate(method?: string | null) {
  return method === "exact-duplicate";
}
const SIMILARITY_REVIEW = 45;
const SIMILARITY_HIGH = 70;
const LEXICAL_REVIEW = 25;
const LEXICAL_HIGH = 50;
const CURRENT_SIMILARITY_METHODS = new Set(["local-embedding", "local-lexical", "exact-duplicate", "none"]);
function similarityThresholds(method?: string | null) {
  return method === "local-lexical"
    ? { review: LEXICAL_REVIEW, high: LEXICAL_HIGH }
    : { review: SIMILARITY_REVIEW, high: SIMILARITY_HIGH };
}
function scoreClass(score: number, method?: string | null) {
  if (isExactDuplicate(method)) return "red";
  const { review, high } = similarityThresholds(method);
  if (score < review) return "green";
  if (score < high) return "yellow";
  return "red";
}
function scoreText(score: number, method?: string | null) {
  if (isExactDuplicate(method)) return "точный дубликат";
  const { review, high } = similarityThresholds(method);
  if (score < review) return "низкое сходство";
  if (score < high) return "проверить";
  return "высокое сходство";
}
function profileClass(score: number) {
  if (score >= 70) return "green";
  if (score >= 45) return "yellow";
  return "red";
}
function profileText(score: number) {
  if (score >= 70) return "хорошее соответствие";
  if (score >= 45) return "проверить профиль";
  return "слабое соответствие";
}
function qualityLabel(topic: Topic) {
  if (topic.quality_state === "blocked") return { label: "✕ Блокирует", cls: "blocked" };
  if (topic.quality_state === "review") return { label: "⚠ Проверить", cls: "review" };
  return { label: "✓ Прошла", cls: "passed" };
}
function scoreValue(score: number, method?: string | null) {
  return isExactDuplicate(method) ? "100/100" : `${Math.round(score)}/100`;
}

function topicSourceLabel(topic: Topic) {
  if (topic.generation_source === "mistral-ai") {
    return { label: "✦ Mistral AI", title: topic.generation_model ? `Сгенерировано моделью ${topic.generation_model}` : "Сгенерировано Mistral AI", cls: "ai" };
  }
  if (topic.generation_source === "local-demo") {
    return { label: "⚙ Демо без ИИ", title: "Тема взята/сформирована локальным демонстрационным генератором без вызова LLM", cls: "demo" };
  }
  return { label: "Источник не зафиксирован", title: "Тема создана в старой версии приложения", cls: "legacy" };
}



export default function Home() {
  const [tab, setTabState] = useState<Tab>("generation");
  const [teachers, setTeachers] = useState<Teacher[]>([]);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [batches, setBatches] = useState<GenerationBatch[]>([]);
  const [activeBatchId, setActiveBatchId] = useState<number | null>(null);
  const activeBatchRef = useRef<number | null>(null);
  const [integrations, setIntegrations] = useState<IntegrationsStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: "ok" | "err"; text: string } | null>(null);
  const [teacherModal, setTeacherModal] = useState<Teacher | "new" | null>(null);
  const [editTopic, setEditTopic] = useState<Topic | null>(null);
  const [filterTeacher, setFilterTeacher] = useState<number | "all">("all");
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [googleUrl, setGoogleUrl] = useState<string | null>(null);
  const [pastDraft, setPastDraft] = useState<Record<number, string>>({});

  const [selectedTeacherIds, setSelectedTeacherIds] = useState<number[]>([]);
  const [generationCounts, setGenerationCounts] = useState<Record<number, number>>({});
  const [generationFocus, setGenerationFocus] = useState("");
  const [generationProgress, setGenerationProgress] = useState<GenerationProgressEvent | null>(null);
  const [teacherSearch, setTeacherSearch] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [deleteTeacherTarget, setDeleteTeacherTarget] = useState<Teacher | null>(null);
  const [batchDeleteConfirmId, setBatchDeleteConfirmId] = useState<number | null>(null);
  const batchDeleteTimer = useRef<number | null>(null);
  const [pastDeleteConfirmId, setPastDeleteConfirmId] = useState<number | null>(null);
  const pastDeleteTimer = useRef<number | null>(null);

  const teacherImportRef = useRef<HTMLInputElement | null>(null);
  const historyImportRef = useRef<HTMLInputElement | null>(null);
  const [historyFile, setHistoryFile] = useState<File | null>(null);
  const [historyPreview, setHistoryPreview] = useState<HistoryImportPreview | null>(null);

  function tabFromLocation(): Tab {
    const value = new URLSearchParams(window.location.search).get("view");
    return value === "teachers" || value === "generation" || value === "topics" || value === "history" || value === "publish"
      ? value
      : "generation";
  }

  function tabUrl(next: Tab) {
    const url = new URL(window.location.href);
    if (next === "generation") url.searchParams.delete("view");
    else url.searchParams.set("view", next);
    return `${url.pathname}${url.search}${url.hash}`;
  }

  function setTab(next: Tab, options?: { replace?: boolean; batchId?: number | null }) {
    const batchId = options?.batchId !== undefined ? options.batchId : activeBatchRef.current;
    if (!options?.replace && next === tab && batchId === activeBatchRef.current) return;
    const state = { vkrNavigation: true, tab: next, batchId };
    if (options?.replace) window.history.replaceState(state, "", tabUrl(next));
    else window.history.pushState(state, "", tabUrl(next));
    setTabState(next);
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  useEffect(() => { activeBatchRef.current = activeBatchId; }, [activeBatchId]);
  useEffect(() => {
    const saved = window.localStorage.getItem("vkr-sidebar-collapsed");
    if (saved === "1") setSidebarCollapsed(true);
  }, []);

  useEffect(() => {
    const initialTab = tabFromLocation();
    setTabState(initialTab);
    window.history.replaceState(
      { vkrNavigation: true, tab: initialTab, batchId: activeBatchRef.current },
      "",
      tabUrl(initialTab),
    );

    const onPopState = (event: PopStateEvent) => {
      const state = event.state as { tab?: Tab; batchId?: number | null } | null;
      const nextTab = state?.tab || tabFromLocation();
      setTabState(nextTab);
      const requestedBatch = state?.batchId;
      if ((nextTab === "topics" || nextTab === "publish") && requestedBatch && requestedBatch !== activeBatchRef.current) {
        activeBatchRef.current = requestedBatch;
        setActiveBatchId(requestedBatch);
        void api.topics(requestedBatch).then(setTopics).catch(() => undefined);
      }
      window.scrollTo({ top: 0, behavior: "auto" });
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  function toggleSidebar() {
    setSidebarCollapsed((current) => {
      const next = !current;
      window.localStorage.setItem("vkr-sidebar-collapsed", next ? "1" : "0");
      return next;
    });
  }

  async function refreshIntegrationStatus() {
    try {
      const result = await api.integrationsStatus();
      setIntegrations(result);
      return result;
    } catch (e) {
      const detail = e instanceof Error ? e.message : "Не удалось проверить интеграции";
      const failed: IntegrationsStatus = {
        mistral: { status: "error", message: "ошибка подключения", detail },
        google_sheets: { status: "error", message: "ошибка подключения", detail },
      };
      setIntegrations(failed);
      return failed;
    }
  }

  async function loadBatchTopics(batchId: number) {
    let rows = await api.topics(batchId);
    const stale = rows.some((topic) =>
      !CURRENT_SIMILARITY_METHODS.has(topic.teacher_similarity_method || "") ||
      !CURRENT_SIMILARITY_METHODS.has(topic.global_similarity_method || "")
    );
    if (stale) {
      try { rows = await api.recalculateBatchSimilarity(batchId); } catch { /* старый набор всё равно можно открыть */ }
    }
    return rows;
  }

  async function refresh(targetBatchId?: number | null) {
    try {
      const [tRaw, bs, cfg] = await Promise.all([api.teachers(), api.batches(), api.settings()]);
      const t = sortTeachers(tRaw);
      const chosen = targetBatchId !== undefined ? targetBatchId : (activeBatchRef.current ?? bs[0]?.id ?? null);
      const tp = chosen ? await loadBatchTopics(chosen) : [];
      setTeachers(t);
      setBatches(bs);
      setAppSettings(cfg);
      setActiveBatchId(chosen);
      setTopics(tp);
      setGenerationFocus((current) => current || cfg.default_generation_focus || "");
      setGenerationCounts((current) => {
        const next = { ...current };
        for (const teacher of t) if (!next[teacher.id]) next[teacher.id] = cfg.default_topic_count || teacher.topic_count || 5;
        return next;
      });
      return true;
    } catch (e) {
      setMessage({ type: "err", text: e instanceof Error ? e.message : "Ошибка загрузки" });
      return false;
    } finally { setLoading(false); }
  }

  useEffect(() => { void refresh(); void refreshIntegrationStatus(); }, []);

  const currentBatch = useMemo(() => batches.find((b) => b.id === activeBatchId) || null, [batches, activeBatchId]);
  const latestBatch = batches[0] || null;
  const approved = topics.filter((t) => t.status === "approved").length;
  const needsAttention = (t: Topic) => {
    if (t.quality_state === "blocked" || t.quality_state === "review") return true;
    const { high } = similarityThresholds(t.global_similarity_method);
    return isExactDuplicate(t.global_similarity_method) || t.global_similarity_score >= high;
  };
  const mistralReady = integrations?.mistral.status === "connected";
  const red = topics.filter((t) => needsAttention(t) && t.status !== "rejected").length;
  const selectedTeachers = teachers.filter((t) => selectedTeacherIds.includes(t.id));
  const requested = selectedTeachers.reduce((sum, t) => sum + (generationCounts[t.id] || 1), 0);
  const filteredTopics = useMemo(() => topics.filter((t) =>
    (filterTeacher === "all" || t.teacher_id === filterTeacher) &&
    (filterStatus === "all" || t.status === filterStatus)
  ), [topics, filterTeacher, filterStatus]);
  const visibleTeachers = useMemo(() => {
    const q = teacherSearch.trim().toLowerCase();
    if (!q) return teachers;
    return teachers.filter((teacher) => [teacher.full_name, teacher.department || "", teacher.position || ""].join(" ").toLowerCase().includes(q));
  }, [teachers, teacherSearch]);

  async function openSettings() {
    setSettingsOpen(true);
    void refreshIntegrationStatus();
    if (appSettings) return;
    try {
      setAppSettings(await api.settings());
    } catch (e) {
      setMessage({ type: "err", text: e instanceof Error ? e.message : "Не удалось загрузить настройки" });
    }
  }

  async function importTeachersFile(file?: File) {
    if (!file) return;
    setBusy("teacher-import");
    try {
      let payload: unknown;
      try { payload = JSON.parse(await file.text()); } catch { throw new Error("Файл не является корректным JSON"); }
      const result = await api.importTeachers(payload);
      await refresh();
      setMessage({ type: "ok", text: `База преподавателей загружена: добавлено ${result.created}, обновлено ${result.updated}. Всего: ${result.total}.` });
    } catch (e) { setMessage({ type: "err", text: e instanceof Error ? e.message : "Не удалось загрузить JSON" }); }
    finally { setBusy(null); if (teacherImportRef.current) teacherImportRef.current.value = ""; }
  }

  async function previewHistoryFile(file?: File) {
    if (!file) return;
    setBusy("history-preview");
    try {
      const preview = await api.previewHistoryXlsx(file);
      setHistoryFile(file);
      setHistoryPreview(preview);
    } catch (e) { setMessage({ type: "err", text: e instanceof Error ? e.message : "Не удалось прочитать XLSX" }); }
    finally { setBusy(null); if (historyImportRef.current) historyImportRef.current.value = ""; }
  }

  async function confirmHistoryImport(file: File, sheets: string[], createMissing: boolean) {
    const result = await api.confirmHistoryXlsx(file, sheets, createMissing);
    setHistoryPreview(null); setHistoryFile(null);
    await refresh();
    setMessage({ type: "ok", text: `Импорт завершён: добавлено ${result.imported} тем, дубликатов пропущено ${result.skipped_duplicates}, неизвестных преподавателей ${result.skipped_unknown}.` });
  }

  async function saveTeacher(payload: TeacherFormPayload) {
    try {
      const saved = teacherModal && teacherModal !== "new" ? await api.updateTeacher(teacherModal.id, payload) : await api.createTeacher(payload);
      setTeachers((current) => sortTeachers(current.some((t) => t.id === saved.id) ? current.map((t) => t.id === saved.id ? saved : t) : [...current, saved]));
      const synced = await refresh();
      setMessage(synced ? { type: "ok", text: "Преподаватель сохранён в общей базе" } : { type: "err", text: "Преподаватель сохранён, но список не удалось обновить" });
    } catch (e) { setMessage({ type: "err", text: e instanceof Error ? e.message : "Ошибка сохранения" }); throw e; }
  }

  async function saveAppSettings(payload: AppSettingsUpdate) {
    try {
      const cfg = await api.updateSettings(payload);
      setAppSettings(cfg);
      await refreshIntegrationStatus();
      setMessage({ type: "ok", text: "Настройки сохранены и подключения проверены" });
    } catch (e) { setMessage({ type: "err", text: e instanceof Error ? e.message : "Ошибка сохранения настроек" }); throw e; }
  }

  const blankIntegration: IntegrationCheck = { status: "unconfigured", message: "не настроен" };

  async function checkMistral(payload: { api_key?: string | null; chat_model?: string | null; embedding_model?: string | null }): Promise<IntegrationCheck> {
    try {
      const result = await api.checkMistral(payload);
      setIntegrations((current) => ({
        mistral: result,
        google_sheets: current?.google_sheets || blankIntegration,
      }));
      return result;
    } catch (e) {
      const result: IntegrationCheck = { status: "error", message: "ошибка подключения", detail: e instanceof Error ? e.message : "Ошибка проверки" };
      setIntegrations((current) => ({
        mistral: result,
        google_sheets: current?.google_sheets || blankIntegration,
      }));
      return result;
    }
  }

  async function checkGoogle(payload: { service_account_json?: string | null; spreadsheet_id?: string | null }): Promise<IntegrationCheck> {
    try {
      const result = await api.checkGoogle(payload);
      setIntegrations((current) => ({
        mistral: current?.mistral || blankIntegration,
        google_sheets: result,
      }));
      return result;
    } catch (e) {
      const result: IntegrationCheck = { status: "error", message: "ошибка подключения", detail: e instanceof Error ? e.message : "Ошибка проверки" };
      setIntegrations((current) => ({
        mistral: current?.mistral || blankIntegration,
        google_sheets: result,
      }));
      return result;
    }
  }

  async function confirmRemoveTeacher(teacher: Teacher) {
    await api.deleteTeacher(teacher.id);
    setSelectedTeacherIds((ids) => ids.filter((x) => x !== teacher.id));
    await refresh();
    setMessage({ type: "ok", text: "Преподаватель удалён" });
  }
  function toggleTeacher(id: number) { setSelectedTeacherIds((ids) => ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]); }
  function selectForGeneration(teacher: Teacher) {
    setSelectedTeacherIds((ids) => ids.includes(teacher.id) ? ids : [...ids, teacher.id]);
    setGenerationCounts((x) => ({ ...x, [teacher.id]: x[teacher.id] || teacher.topic_count || 5 })); setTab("generation");
  }

  async function generateSelected() {
    if (!selectedTeachers.length) { setMessage({ type: "err", text: "Выберите хотя бы одного преподавателя" }); return; }
    const selections = selectedTeachers.map((teacher) => ({
      teacher_id: teacher.id,
      count: Math.max(1, Math.min(10, Number(generationCounts[teacher.id] || 1))),
    }));
    setBusy("generate"); setMessage(null); setGenerationProgress(null);
    try {
      const result = await api.generateSelectedStream(selections, generationFocus, (event) => {
        setGenerationProgress(event);
      });
      const batchId = result.batch_id || null;
      await refresh(batchId);
      if (batchId) activeBatchRef.current = batchId;
      setTab("topics", { batchId });
      setMessage({
        type: "ok",
        text: `Набор #${batchId} создан: ${result.created || 0} тем. Все темы этого запуска получены через Mistral AI; локальный банк не подмешивался. Проверка сходства выполнена.${result.warning ? ` ${result.warning}` : ""}`,
      });
    } catch (e) {
      setMessage({ type: "err", text: e instanceof Error ? e.message : "Ошибка генерации" });
    } finally {
      setBusy(null);
    }
  }

  async function generateDemo() {
    if (!selectedTeachers.length) { setMessage({ type: "err", text: "Выберите хотя бы одного преподавателя" }); return; }
    const selections = selectedTeachers.map((teacher) => ({
      teacher_id: teacher.id,
      count: Math.max(1, Math.min(10, Number(generationCounts[teacher.id] || 1))),
    }));
    setBusy("generate-demo"); setMessage(null); setGenerationProgress(null);
    try {
      const result = await api.generateSelectedDemo(selections, generationFocus);
      const batchId = result.batch_id || null;
      await refresh(batchId);
      if (batchId) activeBatchRef.current = batchId;
      setTab("topics", { batchId });
      setMessage({
        type: "ok",
        text: `Демо-набор #${batchId} создан: ${result.created || 0} тем. Mistral AI не вызывался; все темы помечены как «Демо без ИИ».`,
      });
    } catch (e) {
      setMessage({ type: "err", text: e instanceof Error ? e.message : "Ошибка демо-генерации" });
    } finally {
      setBusy(null);
    }
  }

  async function regenerate(topic: Topic) {
    setBusy(`regen-${topic.id}`);
    try { await api.regenerateTopic(topic.id); await refresh(activeBatchId); setMessage({ type: "ok", text: "Тема перегенерирована и повторно проверена по всей базе" }); }
    catch (e) { setMessage({ type: "err", text: e instanceof Error ? e.message : "Ошибка" }); }
    finally { setBusy(null); }
  }
  async function regenerateRed() {
    const targets = topics.filter((t) => needsAttention(t) && t.status === "draft");
    if (!targets.length) return;
    setBusy("regen-red");
    try { for (const topic of targets) await api.regenerateTopic(topic.id); await refresh(activeBatchId); setMessage({ type: "ok", text: `Перегенерировано тем: ${targets.length}` }); }
    catch (e) { setMessage({ type: "err", text: e instanceof Error ? e.message : "Ошибка" }); }
    finally { setBusy(null); }
  }
  async function saveTopic(title: string) { if (editTopic) { await api.editTopic(editTopic.id, title); await refresh(activeBatchId); } }
  async function status(topic: Topic, value: string) { await api.setTopicStatus(topic.id, value); await refresh(activeBatchId); }
  async function approveAll() {
    setBusy("approve-all");
    try {
      const r = await api.approveAll(activeBatchId);
      await refresh(activeBatchId);
      setMessage({ type: "ok", text: `Утверждено: ${r.approved}. Можно публиковать набор.` });
      setTab("publish", { batchId: activeBatchId });
    } catch (e) {
      setMessage({ type: "err", text: e instanceof Error ? e.message : "Не удалось утвердить темы" });
    } finally {
      setBusy(null);
    }
  }
  async function addPastTopic(teacher: Teacher) {
    const title = (pastDraft[teacher.id] || "").trim(); if (!title) return;
    await api.addPastTopic(teacher.id, title); setPastDraft((x) => ({ ...x, [teacher.id]: "" })); await refresh(activeBatchId);
  }
  async function createGoogleSheet() {
    setBusy("google"); setGoogleUrl(null);
    try { const r = await api.googleSheet(activeBatchId); setGoogleUrl(r.spreadsheet_url); setMessage({ type: "ok", text: `Google-таблица создана. Строк: ${r.rows}` }); }
    catch (e) { setMessage({ type: "err", text: e instanceof Error ? e.message : "Ошибка Google Sheets" }); }
    finally { setBusy(null); }
  }

  async function openBatch(batch: GenerationBatch) {
    setActiveBatchId(batch.id);
    activeBatchRef.current = batch.id;
    setTopics(await loadBatchTopics(batch.id));
    setTab("topics", { batchId: batch.id });
  }

  async function openLatestBatch() {
    if (!latestBatch) {
      setActiveBatchId(null); activeBatchRef.current = null; setTopics([]); setTab("topics", { batchId: null }); return;
    }
    await openBatch(latestBatch);
  }

  function armDeleteBatch(batchId: number) {
    if (batchDeleteConfirmId === batchId) { void deleteBatch(batchId); return; }
    setBatchDeleteConfirmId(batchId);
    if (batchDeleteTimer.current != null) window.clearTimeout(batchDeleteTimer.current);
    batchDeleteTimer.current = window.setTimeout(() => setBatchDeleteConfirmId(null), 4500);
  }

  async function deleteBatch(batchId: number) {
    setBusy(`delete-batch-${batchId}`);
    try {
      const result = await api.deleteBatch(batchId);
      if (activeBatchRef.current === batchId) { activeBatchRef.current = null; setActiveBatchId(null); }
      setBatchDeleteConfirmId(null);
      await refresh();
      setMessage({ type: "ok", text: `Набор удалён. Удалено тем: ${result.deleted_topics}.` });
    } catch (e) { setMessage({ type: "err", text: e instanceof Error ? e.message : "Не удалось удалить набор" }); }
    finally { setBusy(null); }
  }

  async function clearHistory() {
    setBusy("clear-history");
    try {
      const result = await api.clearBatches();
      activeBatchRef.current = null; setActiveBatchId(null); setTopics([]);
      await refresh(null);
      setMessage({ type: "ok", text: `История очищена: ${result.deleted_batches} наборов, ${result.deleted_topics} тем.` });
    } catch (e) { setMessage({ type: "err", text: e instanceof Error ? e.message : "Не удалось очистить историю" }); }
    finally { setBusy(null); }
  }

  function armPastTopicDelete(topic: PastTopic) {
    if (pastDeleteConfirmId === topic.id) { void deletePastTopicConfirmed(topic.id); return; }
    setPastDeleteConfirmId(topic.id);
    if (pastDeleteTimer.current != null) window.clearTimeout(pastDeleteTimer.current);
    pastDeleteTimer.current = window.setTimeout(() => setPastDeleteConfirmId(null), 5000);
  }

  async function deletePastTopicConfirmed(topicId: number) {
    try {
      await api.deletePastTopic(topicId);
      setPastDeleteConfirmId(null);
      await refresh(activeBatchId);
      setMessage({ type: "ok", text: "Тема прошлых лет удалена" });
    } catch (e) {
      setMessage({ type: "err", text: e instanceof Error ? e.message : "Не удалось удалить тему" });
    }
  }


  if (loading) return <div className="loading"><div className="spinner"/><span>Загружаем систему…</span></div>;

  return (
    <main>
      <input ref={teacherImportRef} className="hidden-file-input" type="file" accept=".json,application/json" onChange={(e) => importTeachersFile(e.target.files?.[0])}/>
      <input ref={historyImportRef} className="hidden-file-input" type="file" accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" onChange={(e) => previewHistoryFile(e.target.files?.[0])}/>

      <header className={`topbar ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <div className="sidebar-toggle-zone"><button className="sidebar-toggle" type="button" onClick={toggleSidebar} title={sidebarCollapsed ? "Развернуть боковую панель" : "Свернуть боковую панель"}><span className="sidebar-toggle-icon">{sidebarCollapsed ? "›" : "‹"}</span><span className="sidebar-toggle-text">{sidebarCollapsed ? "Развернуть" : "Свернуть панель"}</span></button></div>
        <div className="program-pill"><span className="dot"/>09.03.01 «Информатика и вычислительная техника»</div><div className="topbar-spacer"/>
      </header>

      <div className={`shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
        <aside className="sidebar">
          <div className="side-caption">Рабочая область</div>
          <button className={tab === "generation" ? "nav active" : "nav"} onClick={() => setTab("generation")}><span>◎</span><div>Новая генерация<small>{selectedTeacherIds.length} выбрано</small></div></button>
          <button className={tab === "topics" || tab === "history" ? "nav active" : "nav"} onClick={() => setTab("topics")}><span>✦</span><div>Темы ВКР<small>{topics.length} в открытом наборе</small></div></button>
          <button className={tab === "publish" ? "nav active" : "nav"} onClick={() => setTab("publish")}><span>↗</span><div>Публикация<small>{approved} утверждено</small></div></button>
          <div className="side-caption second">Интеграции</div>
          <button className="sidebar-settings" type="button" onClick={openSettings} title="Настройки интеграций"><span className="settings-gear">⚙</span><div><b>Настройки</b><small>Mistral и Google Sheets</small></div></button>
          <div className="integration"><i className={integrations?.mistral.status === "connected" ? "on" : integrations?.mistral.status === "error" ? "error" : "off"}/><div><b>Mistral AI</b><small>{integrations?.mistral.message || "проверяем…"}</small></div></div>
          <div className="integration"><i className={integrations?.google_sheets.status === "connected" ? "on" : integrations?.google_sheets.status === "error" ? "error" : "off"}/><div><b>Google Sheets</b><small>{integrations?.google_sheets.message || "проверяем…"}</small></div></div>
        </aside>

        <section className="content">
          {message && <div className={`toast ${message.type}`}>{message.text}<button onClick={() => setMessage(null)}>×</button></div>}
          <div className="stats">
            <div className="stat"><span>В базе</span><strong>{teachers.length}</strong><small>преподавателей</small></div>
            <div className="stat"><span>Выбрано</span><strong>{selectedTeacherIds.length}</strong><small>для новой генерации</small></div>
            <div className="stat"><span>К генерации</span><strong>{requested}</strong><small>тем по выбору</small></div>
            <div className="stat warning"><span>Требуют внимания</span><strong>{red}</strong><small>высокое сходство, точный дубликат или слабое соответствие профилю</small></div>
          </div>

          {tab === "teachers" && <>
            <div className="page-head"><div><div className="eyebrow">Постоянный справочник</div><h1>База преподавателей</h1></div><div className="head-actions">
              <button className="btn secondary" onClick={() => historyImportRef.current?.click()} disabled={busy === "history-preview"}>{busy === "history-preview" ? "Читаем XLSX…" : "▦ Импортировать прошлые ВКР"}</button>
              <a className="btn secondary" href={`${API_URL}/api/teachers/export-json`}>↑ Выгрузить JSON</a>
              <button className="btn secondary" onClick={() => teacherImportRef.current?.click()} disabled={busy === "teacher-import"}>{busy === "teacher-import" ? "Загрузка…" : "↓ Загрузить JSON"}</button>
              <button className="btn secondary" onClick={() => setTeacherModal("new")}>+ Добавить преподавателя</button>
              <button className="btn primary" onClick={() => setTab("generation")} disabled={!teachers.length}>Перейти к выбору →</button>
            </div></div>
            {!teachers.length ? <div className="empty"><div className="empty-icon">👥</div><h3>База пока пустая</h3><button className="btn primary" onClick={() => setTeacherModal("new")}>Добавить преподавателя</button></div> :
            <div className="teacher-grid">{teachers.map((teacher) => <article className="teacher-card" key={teacher.id}>
              <div className="teacher-top"><div className="teacher-avatar">{teacher.full_name.slice(0, 1)}</div><div className="teacher-title"><h3>{teacher.full_name}</h3><span>{[teacher.position, teacher.department].filter(Boolean).join(" · ") || "Должность и кафедра не указаны"}</span></div><div className="catalog-badge">В БАЗЕ</div></div>
              <div className="teacher-meta teacher-meta-simple"><span>Кафедра <b>{teacher.department || "—"}</b></span><span>Должность <b>{teacher.position || "—"}</b></span><span>Одобренных тем <b>{teacher.past_topics.length}</b></span><span>Контакт <b>{teacher.email || teacher.telegram || teacher.phone ? "есть" : "—"}</b></span></div>
              {teacher.research_areas?.length > 0 && <div className="chips">{teacher.research_areas.slice(0, 4).map((x) => <span key={x}>{x}</span>)}</div>}
              <details className="past-box approved-past-box"><summary>Одобренные темы ВКР прошлых лет <span>{teacher.past_topics.length}</span></summary><div className="past-list">{teacher.past_topics.slice(0, 14).map((p) => <div key={p.id} className={pastDeleteConfirmId === p.id ? "delete-armed" : ""}><span>{p.title}</span><button className={pastDeleteConfirmId === p.id ? "past-delete-confirm" : ""} title={pastDeleteConfirmId === p.id ? "Нажмите ещё раз для удаления" : "Удалить тему"} onClick={() => armPastTopicDelete(p)}>{pastDeleteConfirmId === p.id ? "Удалить?" : "×"}</button></div>)}{!teacher.past_topics.length && <p>Тем пока нет.</p>}</div><div className="add-past"><input value={pastDraft[teacher.id] || ""} onChange={(e) => setPastDraft((x) => ({ ...x, [teacher.id]: e.target.value }))} placeholder="Добавить одобренную тему…" onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void addPastTopic(teacher); } }}/><button onClick={() => addPastTopic(teacher)}>+</button></div></details>
              <div className="card-actions"><button onClick={() => setTeacherModal(teacher)}>Настроить</button><button className="select-action" onClick={() => selectForGeneration(teacher)}>+ В генерацию</button><button className="danger-link" onClick={() => setDeleteTeacherTarget(teacher)}>Удалить</button></div>
            </article>)}</div>}
          </>}

          {tab === "generation" && <>
            <div className="page-head"><div><div className="eyebrow">Новый набор тем</div><h1>Выберите преподавателей</h1></div><div className="head-actions"><button className="btn secondary teacher-base-btn" onClick={() => setTab("teachers")}>👥 База преподавателей</button><button className="btn secondary" onClick={() => setSelectedTeacherIds(teachers.map((t) => t.id))}>Выбрать всех</button><button className="btn ghost" onClick={() => setSelectedTeacherIds([])}>Сбросить</button></div></div>
            {!mistralReady && <div className="ai-required-banner"><div><b>Mistral AI не подключён — AI-генерация отключена</b><span>Основная кнопка больше не подставляет готовые темы. Для проверки интерфейса можно отдельно запустить «Демо без ИИ».</span></div><button className="btn secondary" onClick={() => setSettingsOpen(true)}>Подключить Mistral</button></div>}
            {!teachers.length ? <div className="empty"><h3>Сначала заполните базу</h3><button className="btn primary" onClick={() => setTab("teachers")}>К базе</button></div> : <div className="generation-layout">
              <div className="generation-panel"><div className="selection-toolbar"><input value={teacherSearch} onChange={(e) => setTeacherSearch(e.target.value)} placeholder="Поиск по ФИО, кафедре или должности…"/><span>{selectedTeacherIds.length} из {teachers.length} выбрано</span></div><div className="selection-list">{visibleTeachers.map((teacher) => { const checked = selectedTeacherIds.includes(teacher.id); return <div className={`selection-row ${checked ? "selected" : ""}`} key={teacher.id}><label className="teacher-check"><input type="checkbox" checked={checked} onChange={() => toggleTeacher(teacher.id)}/><span className="check-mark">✓</span></label><div className="selection-person"><b>{teacher.full_name}</b><small>{[teacher.position, teacher.department].filter(Boolean).join(" · ") || "данные не заполнены"}</small></div><label className="count-control"><span>Количество тем</span><input type="number" min={1} max={10} disabled={!checked} value={generationCounts[teacher.id] || 5} onChange={(e) => setGenerationCounts((x) => ({ ...x, [teacher.id]: Math.max(1, Math.min(10, Number(e.target.value) || 1)) }))}/></label></div>; })}</div></div>
              <aside className="generation-summary"><div className="summary-kicker">Текущий выбор</div><strong>{requested}</strong><span>тем запрошено (Quality Gate может оставить меньше)</span><div className="summary-divider"/><p><b>{selectedTeacherIds.length}</b> преподавателей.</p><div className="batch-generation-hint"><b>Массовая AI-генерация до 100+ тем</b><span>Каждый преподаватель обрабатывается отдельным коротким AI-запросом до 10 тем. Например, 10 преподавателей × 10 тем = 10 последовательных AI-запросов с прогрессом после каждого преподавателя.</span><small>V34: для скорости Mistral возвращает только короткий список названий тем — без длинных rationale/keywords в structured-ответе. По названию всё равно должно быть ясно, какой программный продукт создаётся, для какой задачи и с какими функциями. Профиль, прошлые темы и дубли проверяет локальный Quality Gate.</small></div><label className="focus-field"><span>Фокус генерации <i>необязательно</i></span><textarea rows={6} value={generationFocus} onChange={(e) => setGenerationFocus(e.target.value)} placeholder="Например: компьютерное зрение; веб-системы; открытые датасеты…"/></label>{busy === "generate" && generationProgress && <div className="generation-progress"><div className="generation-progress-head"><b>{generationProgress.stage === "similarity" ? "Проверка сходства" : "Генерация тем"}</b><span>{generationProgress.created || 0} / {generationProgress.total || requested}</span></div><div className="generation-progress-track"><i style={{ width: `${Math.min(100, Math.round(((generationProgress.created || 0) / Math.max(1, generationProgress.total || requested)) * 100))}%` }}/></div><small>{generationProgress.message || (generationProgress.stage === "similarity" ? "Проверяем темы…" : `Преподаватель ${generationProgress.completed_groups || 0} из ${generationProgress.total_groups || "…"}`)}</small></div>}<button className="btn primary glow generation-button" onClick={generateSelected} disabled={!selectedTeacherIds.length || busy === "generate" || busy === "generate-demo" || !mistralReady} title={!mistralReady ? "Сначала подключите и проверьте Mistral AI" : "Генерация только через Mistral AI"}>{busy === "generate" ? (generationProgress?.stage === "similarity" ? "Проверяем сходство…" : `AI-генерация ${generationProgress?.created || 0}/${generationProgress?.total || requested}…`) : `✦ Mistral AI: ${requested || 0} тем`}</button><button className="btn secondary demo-generation-button" onClick={generateDemo} disabled={!selectedTeacherIds.length || busy === "generate" || busy === "generate-demo"}>{busy === "generate-demo" ? "Демо-генерация…" : `⚙ Демо без ИИ: ${requested || 0} тем`}</button></aside>
            </div>}
          </>}

          {tab === "topics" && <>
            <div className="page-head"><div><div className="eyebrow">Открытый набор {currentBatch ? `#${currentBatch.id}` : ""}</div><h1>Сгенерированные темы</h1>{currentBatch && currentBatch.id !== latestBatch?.id && <p className="history-open-note">Открыт архивный набор от {new Date(currentBatch.created_at).toLocaleString("ru-RU")}.</p>}</div><div className="head-actions"><button className="btn secondary" onClick={() => setTab("history")}>◷ История наборов</button><button className="btn secondary" onClick={regenerateRed} disabled={busy === "regen-red" || red === 0}>↻ Перегенерировать красные</button><button className="btn primary" onClick={approveAll} disabled={!topics.some((t) => t.status === "draft")}>{busy === "approve-all" ? "Утверждаем…" : "✓ Утвердить все"}</button></div></div>
            {currentBatch?.focus && <div className="batch-focus"><b>Фокус:</b> {currentBatch.focus}</div>}
            <div className="toolbar"><select value={filterTeacher} onChange={(e) => setFilterTeacher(e.target.value === "all" ? "all" : Number(e.target.value))}><option value="all">Все преподаватели</option>{teachers.map((t) => <option value={t.id} key={t.id}>{t.full_name}</option>)}</select><select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}><option value="all">Все статусы</option><option value="draft">Черновики</option><option value="approved">Утверждённые</option><option value="rejected">Отклонённые</option></select><div className="legend"><span><i className="sgreen"/>низкий риск</span><span><i className="syellow"/>проверить</span><span><i className="sred"/>высокий риск</span><small title="Локальные embeddings: пороги 45/70; lexical fallback: 25/50">пороги зависят от метода</small></div></div>
            {!topics.length ? <div className="empty"><div className="empty-icon">✦</div><h3>В наборе тем нет</h3><button className="btn primary" onClick={() => setTab("generation")}>Новая генерация</button></div> : <div className="topic-table-wrap"><table className="topic-table dual-sim-table"><thead><tr><th>Преподаватель</th><th>Конкретная тема ВКР</th><th title="Локальная постпроверка темы по research_areas и истории преподавателя">Профиль</th><th>Quality Gate</th><th title="Только исторические темы этого преподавателя; исходная готовая тема исключается">Сходство с прошлыми</th><th title="Другие темы этого же текущего набора">Внутри набора</th><th title="Историческая база ВКР + другие темы текущего набора">Сходство по всей базе</th><th>Статус</th><th></th></tr></thead><tbody>{filteredTopics.map((topic) => <tr key={topic.id} className={topic.status === "rejected" ? "muted-row" : ""}><td><b>{topic.teacher_name}</b><small>{topic.teacher_contact || "контакт не указан"}</small></td><td><div className="topic-source-line"><span className={`topic-source-badge ${topicSourceLabel(topic).cls}`} title={topicSourceLabel(topic).title}>{topicSourceLabel(topic).label}</span>{topic.generation_model && topic.generation_source === "mistral-ai" && <small>{topic.generation_model}</small>}</div><div className="topic-title">{topic.title}{topic.manual_edit && <span className="manual">изменено</span>}</div>{topic.rationale && <div className="rationale">{topic.rationale}</div>}{topic.teacher_closest_topic && <details className="closest"><summary>Ближайшая прошлая тема преподавателя</summary><p>{topic.teacher_closest_topic}</p></details>}{topic.global_closest_topic && <details className="closest"><summary>Ближайшая тема по всей базе</summary><p>{topic.global_closest_teacher && <b>{topic.global_closest_teacher}: </b>}{topic.global_closest_topic}</p></details>}</td><td><div className={`score ${profileClass(topic.profile_relevance_score)}`}><b>{Math.round(topic.profile_relevance_score)}/100</b><span>{profileText(topic.profile_relevance_score)}</span></div>{topic.matched_research_areas?.length > 0 && <small className="profile-areas">{topic.matched_research_areas.join(" · ")}</small>}{topic.foreign_profile_directions?.length > 0 && <small className="profile-areas profile-warning">чужое направление: {topic.foreign_profile_directions.join(" · ")}</small>}</td><td><span className={`quality-gate ${qualityLabel(topic).cls}`}>{qualityLabel(topic).label}</span>{topic.quality_reasons?.length > 0 && <details className="closest"><summary>Причины</summary><p>{topic.quality_reasons.join("; ")}</p></details>}</td><td><div className={`score ${scoreClass(topic.teacher_similarity_score, topic.teacher_similarity_method)}`} title={`Метод: ${topic.teacher_similarity_method}`}><b>{scoreValue(topic.teacher_similarity_score, topic.teacher_similarity_method)}</b><span>{scoreText(topic.teacher_similarity_score, topic.teacher_similarity_method)}</span></div></td><td><div className={`score ${scoreClass(topic.batch_similarity_score, topic.batch_similarity_method)}`} title={`Метод: ${topic.batch_similarity_method}`}><b>{scoreValue(topic.batch_similarity_score, topic.batch_similarity_method)}</b><span>{scoreText(topic.batch_similarity_score, topic.batch_similarity_method)}</span></div>{topic.batch_closest_topic && <details className="closest"><summary>Ближайшая в наборе</summary><p>{topic.batch_closest_teacher && <b>{topic.batch_closest_teacher}: </b>}{topic.batch_closest_topic}</p></details>}</td><td><div className={`score ${scoreClass(topic.global_similarity_score, topic.global_similarity_method)}`} title={`Метод: ${topic.global_similarity_method}`}><b>{scoreValue(topic.global_similarity_score, topic.global_similarity_method)}</b><span>{scoreText(topic.global_similarity_score, topic.global_similarity_method)}</span></div></td><td><span className={`status ${topic.status}`}>{topic.status === "draft" ? "Черновик" : topic.status === "approved" ? "Утверждена" : "Отклонена"}</span></td><td><div className="row-actions"><button title="Редактировать" onClick={() => setEditTopic(topic)}>✎</button><button title="Перегенерировать" onClick={() => regenerate(topic)} disabled={busy === `regen-${topic.id}`}>{busy === `regen-${topic.id}` ? "…" : "↻"}</button>{topic.status !== "approved" ? <button className="approve" title="Утвердить" onClick={() => status(topic, "approved")}>✓</button> : <button title="Вернуть в черновик" onClick={() => status(topic, "draft")}>↶</button>}<button className="reject" title="Отклонить" onClick={() => status(topic, "rejected")}>×</button></div></td></tr>)}</tbody></table></div>}
            <div className="bottom-actions"><span>Утверждено <b>{approved}</b> из {topics.filter((t) => t.status !== "rejected").length}</span><button className="btn primary" onClick={() => setTab("publish")}>Публикация →</button></div>
          </>}

          {tab === "history" && <>
            <div className="page-head"><div><div className="eyebrow">Темы ВКР · архив</div><h1>История наборов</h1></div><div className="head-actions"><button className="btn secondary" onClick={openLatestBatch} disabled={!latestBatch}>Последний набор</button><button className="btn primary" onClick={() => setTab("generation")}>+ Новый набор</button></div></div>
            {!batches.length ? <div className="empty"><h3>История пока пустая</h3><p>После первой генерации здесь появятся сохранённые наборы.</p></div> : <>
              <div className="batch-list">{batches.map((batch) => <div key={batch.id} className={`batch-card ${batch.id === activeBatchId ? "active" : ""}`}><div className="batch-number">Набор #{batch.id}</div><div className="batch-date">{new Date(batch.created_at).toLocaleString("ru-RU")}</div><div className="batch-metrics"><span><b>{batch.topic_count}</b> тем</span><span><b>{batch.teacher_count}</b> преподавателей</span><span><b>{batch.approved_count}</b> утверждено</span><span className={batch.attention_count ? "danger" : ""}><b>{batch.attention_count}</b> требуют внимания</span></div>{batch.focus && <p>{batch.focus}</p>}<div className="batch-actions"><button className="btn secondary" onClick={() => openBatch(batch)}>Открыть</button><a className={`btn secondary ${batch.approved_count ? "" : "disabled"}`} href={batch.approved_count ? `${API_URL}/api/export/xlsx?batch_id=${batch.id}` : undefined}>↓ Выгрузить XLSX</a><button className={`btn batch-delete-btn ${batchDeleteConfirmId === batch.id ? "armed" : ""}`} onClick={() => armDeleteBatch(batch.id)} disabled={busy === `delete-batch-${batch.id}`}>{busy === `delete-batch-${batch.id}` ? "Удаляем…" : batchDeleteConfirmId === batch.id ? "Подтвердить удаление" : "Удалить"}</button></div></div>)}</div>
              <div className="history-clear-zone"><div><b>Очистить всю историю</b><span>Будут удалены {batches.length} наборов и все сгенерированные в них темы. База преподавателей и прошлые ВКР останутся.</span></div><HoldToClearButton disabled={busy === "clear-history" || !batches.length} onConfirm={clearHistory}/></div>
            </>}
          </>}

          {tab === "publish" && <>
            <div className="page-head"><div><div className="eyebrow">Публикация {currentBatch ? `набора #${currentBatch.id}` : ""}</div><h1>Публикация</h1></div></div>
            <div className="publish-grid"><div className="publish-card main"><div className="publish-icon">▦</div><h2>Итоговый список тем</h2><div className="publish-stats"><div><strong>{approved}</strong><span>тем</span></div><div><strong>{new Set(topics.filter((t) => t.status === "approved").map((t) => t.teacher_id)).size}</strong><span>преподавателей</span></div></div><div className="export-actions"><a className={`btn secondary ${approved ? "" : "disabled"}`} href={approved ? `${API_URL}/api/export/xlsx${activeBatchId ? `?batch_id=${activeBatchId}` : ""}` : undefined}>↓ Скачать XLSX</a><button className="btn google" onClick={createGoogleSheet} disabled={!approved || busy === "google"}>{busy === "google" ? "Создаём…" : "Создать Google-таблицу ↗"}</button></div>{googleUrl && <a href={googleUrl} target="_blank" className="sheet-link">Открыть Google-таблицу ↗</a>}</div><div className="publish-card"><h3>Структура таблицы</h3><div className="mini-table"><div className="mini-head"><span>Преподаватель</span><span>Тема</span><span>ФИО студента</span></div>{topics.filter((t) => t.status === "approved").slice(0, 4).map((t) => <div className="mini-row" key={t.id}><span>{t.teacher_name}</span><span>{t.title}</span><span className="blank">Студент впишет себя</span></div>)}</div></div></div>
          </>}
        </section>
      </div>

      {teacherModal && <TeacherModal teacher={teacherModal === "new" ? null : teacherModal} onClose={() => setTeacherModal(null)} onSave={saveTeacher}/>} 
      {editTopic && <TopicEditModal topic={editTopic} onClose={() => setEditTopic(null)} onSave={saveTopic}/>} 
      {settingsOpen && <SettingsModal settings={appSettings ?? DEFAULT_APP_SETTINGS} integrations={integrations} onClose={() => setSettingsOpen(false)} onSave={saveAppSettings} onCheckMistral={checkMistral} onCheckGoogle={checkGoogle}/>} 
      {historyPreview && historyFile && <HistoryImportModal file={historyFile} preview={historyPreview} onClose={() => { setHistoryPreview(null); setHistoryFile(null); }} onConfirm={confirmHistoryImport}/>} 
      {deleteTeacherTarget && <TeacherDeleteModal teacher={deleteTeacherTarget} onClose={() => setDeleteTeacherTarget(null)} onConfirm={() => confirmRemoveTeacher(deleteTeacherTarget)}/>} 
    </main>
  );
}
