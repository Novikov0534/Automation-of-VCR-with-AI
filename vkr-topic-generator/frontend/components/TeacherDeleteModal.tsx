"use client";

import { useEffect, useState } from "react";
import type { Teacher } from "../lib/api";

export default function TeacherDeleteModal({
  teacher,
  onClose,
  onConfirm,
}: {
  teacher: Teacher;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!armed) return;
    const timer = window.setTimeout(() => setArmed(false), 4500);
    return () => window.clearTimeout(timer);
  }, [armed]);

  async function confirm() {
    if (!armed) {
      setArmed(true);
      return;
    }
    setBusy(true);
    try {
      await onConfirm();
      onClose();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal small delete-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="eyebrow danger-eyebrow">Удаление профиля</div>
            <h2>Удалить преподавателя?</h2>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Закрыть">×</button>
        </div>
        <div className="delete-person-card">
          <b>{teacher.full_name}</b>
          <span>{[teacher.position, teacher.department].filter(Boolean).join(" · ") || "Должность и кафедра не указаны"}</span>
        </div>
        <p className="delete-warning-text">В профиле сохранено <b>{teacher.past_topics.length}</b> тем ВКР прошлых лет. При удалении профиль и связанные с ним данные будут удалены.</p>
        <div className="modal-actions">
          <button type="button" className="btn ghost" onClick={onClose}>Отмена</button>
          <button type="button" className={`btn ${armed ? "danger-confirm" : "secondary"}`} onClick={confirm} disabled={busy}>
            {busy ? "Удаляем…" : armed ? "Подтвердить удаление" : "Удалить"}
          </button>
        </div>
      </div>
    </div>
  );
}
