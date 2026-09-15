"use client";
import { FormEvent, useState } from "react";
import type { Topic } from "../lib/api";

export default function TopicEditModal({ topic, onClose, onSave }: { topic: Topic; onClose: () => void; onSave: (title: string) => Promise<void> }) {
  const [title, setTitle] = useState(topic.title);
  const [busy, setBusy] = useState(false);
  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    try { await onSave(title.trim()); onClose(); } finally { setBusy(false); }
  }
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal small" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head"><div><div className="eyebrow">Ручная корректировка</div><h2>Изменить тему</h2></div><button className="icon-btn" onClick={onClose}>×</button></div>
        <form onSubmit={submit}>
          <label className="field"><span>Формулировка ВКР</span><textarea rows={5} value={title} onChange={(e) => setTitle(e.target.value)} required /></label>
          <div className="hint">После сохранения система заново рассчитает похожесть с прошлыми и уже созданными темами.</div>
          <div className="modal-actions"><button type="button" className="btn ghost" onClick={onClose}>Отмена</button><button className="btn primary" disabled={busy}>{busy ? "Проверяем…" : "Сохранить"}</button></div>
        </form>
      </div>
    </div>
  );
}
