"use client";

import { useMemo, useState } from "react";
import type { HistoryImportPreview } from "../lib/api";

export default function HistoryImportModal({
  file,
  preview,
  onClose,
  onConfirm,
}: {
  file: File;
  preview: HistoryImportPreview;
  onClose: () => void;
  onConfirm: (file: File, sheets: string[], createMissing: boolean) => Promise<void>;
}) {
  const [selectedSheets, setSelectedSheets] = useState<string[]>(preview.sheets.map((s) => s.name));
  const [createMissing, setCreateMissing] = useState(false);
  const [busy, setBusy] = useState(false);

  const visible = useMemo(
    () => preview.items.filter((item) => selectedSheets.includes(item.sheet)).slice(0, 120),
    [preview.items, selectedSheets],
  );

  function toggleSheet(name: string) {
    setSelectedSheets((current) => current.includes(name) ? current.filter((x) => x !== name) : [...current, name]);
  }

  async function confirmImport() {
    if (!selectedSheets.length) return;
    setBusy(true);
    try { await onConfirm(file, selectedSheets, createMissing); }
    finally { setBusy(false); }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal history-import-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div><div className="eyebrow">Импорт истории</div><h2>Проверка XLSX перед загрузкой</h2></div>
          <button className="icon-btn" onClick={onClose} aria-label="Закрыть">×</button>
        </div>

        <div className="import-file-name">{preview.filename}</div>
        <div className="import-kpis">
          <div><b>{preview.total_detected}</b><span>тем найдено</span></div>
          <div><b>{preview.matched}</b><span>сопоставлено</span></div>
          <div><b>{preview.unknown}</b><span>неизвестных преподавателей</span></div>
          <div><b>{preview.duplicates}</b><span>уже есть в базе</span></div>
        </div>

        <div className="import-section-title">Какие листы импортировать</div>
        <div className="sheet-picker">
          {preview.sheets.map((sheet) => <label key={sheet.name} className={selectedSheets.includes(sheet.name) ? "sheet-choice selected" : "sheet-choice"}>
            <input type="checkbox" checked={selectedSheets.includes(sheet.name)} onChange={() => toggleSheet(sheet.name)} />
            <div><b>{sheet.name}</b><span>{sheet.detected_rows} тем · {sheet.matched_rows} сопоставлено · {sheet.unknown_rows} неизвестно</span></div>
          </label>)}
        </div>

        <label className="import-option">
          <input type="checkbox" checked={createMissing} onChange={(e) => setCreateMissing(e.target.checked)} />
          <div><b>Создавать отсутствующих преподавателей</b><span>Только если в XLSX указано полное ФИО. Если выключено, их темы будут пропущены.</span></div>
        </label>

        <div className="import-section-title">Предпросмотр</div>
        <div className="import-preview-table-wrap">
          <table className="import-preview-table">
            <thead><tr><th>Лист / строка</th><th>Преподаватель</th><th>Тема</th><th>Результат</th></tr></thead>
            <tbody>{visible.map((item, index) => <tr key={`${item.sheet}-${item.row}-${index}`}>
              <td>{item.sheet}<small>стр. {item.row}</small></td>
              <td>{item.teacher_name}<small>{item.matched_teacher_name || "нет в базе"}</small></td>
              <td>{item.title}</td>
              <td>{item.already_exists ? <span className="import-state duplicate">дубликат</span> : item.matched_teacher_id ? <span className="import-state matched">готово</span> : <span className="import-state unknown">не найден</span>}</td>
            </tr>)}</tbody>
          </table>
        </div>
        {preview.items.filter((x) => selectedSheets.includes(x.sheet)).length > visible.length && <div className="import-truncated">В предпросмотре показаны первые {visible.length} строк. Импорт обработает все выбранные листы.</div>}


        <div className="modal-actions">
          <button className="btn ghost" onClick={onClose}>Отмена</button>
          <button className="btn primary" onClick={confirmImport} disabled={!selectedSheets.length || busy}>{busy ? "Импортируем…" : "Подтвердить импорт"}</button>
        </div>
      </div>
    </div>
  );
}
