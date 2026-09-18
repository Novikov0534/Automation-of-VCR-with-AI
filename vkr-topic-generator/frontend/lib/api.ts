export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "").replace(/\/$/, "");

export type PastTopic = {
  id: number;
  title: string;
  year?: number | null;
  source_file?: string | null;
  source_sheet?: string | null;
  source_row?: number | null;
  source_project?: string | null;
  is_reference: boolean;
};

export type Teacher = {
  id: number;
  full_name: string;
  topic_count: number;
  department?: string | null;
  position?: string | null;
  email?: string | null;
  telegram?: string | null;
  phone?: string | null;
  research_areas: string[];
  orcid?: string | null;
  scopus_id?: string | null;
  past_topics: PastTopic[];
  created_at: string;
  updated_at: string;
};

export type TeacherImportResult = { created: number; updated: number; total: number };

export type Topic = {
  id: number;
  teacher_id: number;
  teacher_name: string;
  teacher_contact: string;
  title: string;
  rationale?: string | null;
  keywords: string[];
  generation_source: string;
  generation_model?: string | null;
  profile_relevance_score: number;
  matched_research_areas: string[];
  profile_directions: string[];
  foreign_profile_directions: string[];
  profile_relevance_method: string;
  similarity_score: number;
  closest_past_topic?: string | null;
  teacher_similarity_score: number;
  teacher_closest_topic?: string | null;
  global_similarity_score: number;
  global_closest_topic?: string | null;
  global_closest_teacher?: string | null;
  batch_similarity_score: number;
  batch_closest_topic?: string | null;
  batch_closest_teacher?: string | null;
  similarity_method: string;
  teacher_similarity_method: string;
  global_similarity_method: string;
  batch_similarity_method: string;
  quality_state: "passed" | "review" | "blocked";
  quality_reasons: string[];
  status: "draft" | "approved" | "rejected";
  manual_edit: boolean;
  created_at: string;
  updated_at: string;
};

export type GenerationSelection = { teacher_id: number; count: number };

export type GenerationProgressEvent = {
  type: "start" | "progress" | "heartbeat" | "final" | "error";
  batch_id?: number | null;
  created?: number;
  total?: number;
  completed_groups?: number;
  total_groups?: number;
  stage?: "generation" | "similarity" | "done";
  mode?: string;
  warning?: string | null;
  message?: string;
};

export type AppSettings = {
  mistral_configured: boolean;
  mistral_key_hint?: string | null;
  mistral_chat_model: string;
  mistral_embedding_model: string;
  google_service_account_configured: boolean;
  google_service_account_hint?: string | null;
  google_share_with_email?: string | null;
  google_spreadsheet_id?: string | null;
  default_topic_count: number;
  default_generation_focus?: string | null;
  similarity_green_max: number;
  similarity_yellow_max: number;
  similarity_calibrated: boolean;
};

export type AppSettingsUpdate = {
  mistral_api_key?: string | null;
  mistral_chat_model?: string | null;
  mistral_embedding_model?: string | null;
  google_service_account_json?: string | null;
  google_share_with_email?: string | null;
  google_spreadsheet_id?: string | null;
  default_topic_count?: number | null;
  default_generation_focus?: string | null;
  similarity_green_max?: number | null;
  similarity_yellow_max?: number | null;
};


export type IntegrationCheck = {
  status: "unconfigured" | "connected" | "error";
  message: string;
  detail?: string | null;
};

export type IntegrationsStatus = {
  mistral: IntegrationCheck;
  google_sheets: IntegrationCheck;
};

export type HistoryDeleteResult = { deleted_batches: number; deleted_topics: number };
export type GenerationBatch = {
  id: number;
  focus?: string | null;
  selections: GenerationSelection[];
  topic_count: number;
  teacher_count: number;
  approved_count: number;
  attention_count: number;
  created_at: string;
};

export type HistoryImportItem = {
  sheet: string;
  row: number;
  teacher_name: string;
  title: string;
  year?: number | null;
  project?: string | null;
  matched_teacher_id?: number | null;
  matched_teacher_name?: string | null;
  match_type: string;
  already_exists: boolean;
  recommended_reference: boolean;
};

export type HistoryImportPreview = {
  filename: string;
  total_detected: number;
  matched: number;
  unknown: number;
  duplicates: number;
  recommended_reference: number;
  sheets: { name: string; detected_rows: number; matched_rows: number; unknown_rows: number }[];
  items: HistoryImportItem[];
};

export type HistoryImportResult = {
  imported: number;
  skipped_duplicates: number;
  skipped_unknown: number;
  created_teachers: number;
  reference_count: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
      cache: "no-store",
    });
  } catch {
    throw new Error("Не удалось связаться с сервером приложения. Обновите страницу; если ошибка повторится, проверьте контейнер backend.");
  }
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      message = data.detail || message;
      if (Array.isArray(message)) message = JSON.stringify(message);
    } catch {}
    throw new Error(String(message));
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

async function formRequest<T>(path: string, form: FormData): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { method: "POST", body: form, cache: "no-store" });
  } catch {
    throw new Error("Не удалось связаться с сервером приложения. Обновите страницу; если ошибка повторится, проверьте контейнер backend.");
  }
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message = (await response.json()).detail || message; } catch {}
    throw new Error(String(message));
  }
  return response.json();
}

export const api = {
  health: () => request<{ ok: boolean; mistral_configured: boolean; google_sheets_configured: boolean; demo_mode: boolean }>("/api/health"),
  integrationsStatus: () => request<IntegrationsStatus>("/api/integrations/status"),
  checkMistral: (payload: { api_key?: string | null; chat_model?: string | null; embedding_model?: string | null }) => request<IntegrationCheck>("/api/integrations/check/mistral", { method: "POST", body: JSON.stringify(payload) }),
  checkGoogle: (payload: { service_account_json?: string | null; spreadsheet_id?: string | null }) => request<IntegrationCheck>("/api/integrations/check/google", { method: "POST", body: JSON.stringify(payload) }),
  settings: () => request<AppSettings>("/api/settings"),
  updateSettings: (payload: AppSettingsUpdate) => request<AppSettings>("/api/settings", { method: "PUT", body: JSON.stringify(payload) }),

  researchAreas: () => request<{ items: string[] }>("/api/research-areas"),
  teachers: () => request<Teacher[]>("/api/teachers"),
  createTeacher: (payload: unknown) => request<Teacher>("/api/teachers", { method: "POST", body: JSON.stringify(payload) }),
  updateTeacher: (id: number, payload: unknown) => request<Teacher>(`/api/teachers/${id}`, { method: "PUT", body: JSON.stringify(payload) }),
  deleteTeacher: (id: number) => request<void>(`/api/teachers/${id}`, { method: "DELETE" }),
  addPastTopic: (id: number, title: string, year?: number) => request<Teacher>(`/api/teachers/${id}/past-topics`, { method: "POST", body: JSON.stringify({ title, year }) }),
  deletePastTopic: (id: number) => request<void>(`/api/past-topics/${id}`, { method: "DELETE" }),
  importTeachers: (payload: unknown) => request<TeacherImportResult>("/api/teachers/import-json", { method: "POST", body: JSON.stringify(payload) }),

  previewHistoryXlsx: (file: File) => {
    const form = new FormData(); form.append("file", file);
    return formRequest<HistoryImportPreview>("/api/history-import/preview", form);
  },
  confirmHistoryXlsx: (file: File, sheets: string[], createMissing: boolean) => {
    const form = new FormData();
    form.append("file", file);
    form.append("selected_sheets", JSON.stringify(sheets));
    form.append("create_missing_teachers", String(createMissing));
    form.append("refresh_reference", "true");
    return formRequest<HistoryImportResult>("/api/history-import/confirm", form);
  },

  latestBatch: () => request<GenerationBatch | null>("/api/generation-batches/latest"),
  batches: () => request<GenerationBatch[]>("/api/generation-batches"),
  deleteBatch: (id: number) => request<HistoryDeleteResult>(`/api/generation-batches/${id}`, { method: "DELETE" }),
  clearBatches: () => request<HistoryDeleteResult>("/api/generation-batches", { method: "DELETE" }),
  recalculateBatchSimilarity: (id: number) => request<Topic[]>(`/api/generation-batches/${id}/recalculate-similarity`, { method: "POST" }),
  topics: (batchId?: number | null) => request<Topic[]>(`/api/topics${batchId ? `?batch_id=${batchId}` : ""}`),
  generateSelected: (selections: GenerationSelection[], focus?: string) => request<{ created: number; mode: string; topics: Topic[]; warning?: string | null; batch_id?: number | null }>("/api/generate/selected", {
    method: "POST", body: JSON.stringify({ selections, focus: focus?.trim() || null }),
  }),
  generateSelectedDemo: (selections: GenerationSelection[], focus?: string) => request<{ created: number; mode: string; topics: Topic[]; warning?: string | null; batch_id?: number | null }>("/api/generate/selected/demo", {
    method: "POST", body: JSON.stringify({ selections, focus: focus?.trim() || null }),
  }),
  generateSelectedStream: async (
    selections: GenerationSelection[],
    focus: string | undefined,
    onEvent: (event: GenerationProgressEvent) => void,
  ): Promise<GenerationProgressEvent> => {
    let response: Response;
    try {
      response = await fetch(`${API_URL}/api/generate/selected/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selections, focus: focus?.trim() || null }),
        cache: "no-store",
      });
    } catch {
      throw new Error("Не удалось связаться с сервером приложения во время генерации.");
    }
    if (!response.ok || !response.body) {
      throw new Error(`Не удалось запустить генерацию: HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalEvent: GenerationProgressEvent | null = null;

    const processLine = (line: string) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      const event = JSON.parse(trimmed) as GenerationProgressEvent;
      onEvent(event);
      if (event.type === "error") throw new Error(event.message || "Ошибка генерации");
      if (event.type === "final") finalEvent = event;
    };

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) processLine(line);
      if (done) break;
    }
    if (buffer.trim()) processLine(buffer);
    if (!finalEvent) throw new Error("Сервер завершил поток без итогового результата");
    return finalEvent;
  },
  regenerateTopic: (id: number) => request<Topic>(`/api/topics/${id}/regenerate`, { method: "POST" }),
  editTopic: (id: number, title: string) => request<Topic>(`/api/topics/${id}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  setTopicStatus: (id: number, status: string) => request<Topic>(`/api/topics/${id}/status`, { method: "PATCH", body: JSON.stringify({ status }) }),
  approveAll: (batchId?: number | null) => request<{ approved: number }>(`/api/topics/approve-all${batchId ? `?batch_id=${batchId}` : ""}`, { method: "POST" }),
  deleteTopic: (id: number) => request<void>(`/api/topics/${id}`, { method: "DELETE" }),

  googleSheet: (batchId?: number | null) => request<{ spreadsheet_id: string; spreadsheet_url: string; rows: number }>(`/api/export/google-sheet${batchId ? `?batch_id=${batchId}` : ""}`, { method: "POST" }),


};
