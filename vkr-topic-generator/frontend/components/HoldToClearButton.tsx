"use client";

import { useRef, useState } from "react";
import type { CSSProperties } from "react";

export default function HoldToClearButton({
  disabled,
  onConfirm,
}: {
  disabled?: boolean;
  onConfirm: () => Promise<void>;
}) {
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const timer = useRef<number | null>(null);
  const startAt = useRef(0);

  function stop() {
    if (timer.current != null) window.clearInterval(timer.current);
    timer.current = null;
    if (!busy) setProgress(0);
  }

  function start() {
    if (disabled || busy) return;
    stop();
    startAt.current = performance.now();
    timer.current = window.setInterval(async () => {
      const value = Math.min(100, ((performance.now() - startAt.current) / 1100) * 100);
      setProgress(value);
      if (value >= 100) {
        if (timer.current != null) window.clearInterval(timer.current);
        timer.current = null;
        setBusy(true);
        try {
          await onConfirm();
        } finally {
          setBusy(false);
          setProgress(0);
        }
      }
    }, 35);
  }

  return (
    <button
      type="button"
      className="hold-clear-btn"
      disabled={disabled || busy}
      onPointerDown={start}
      onPointerUp={stop}
      onPointerLeave={stop}
      onPointerCancel={stop}
      style={{ "--hold-progress": `${progress}%` } as CSSProperties}
      title="Удерживайте кнопку примерно секунду"
    >
      <span>{busy ? "Очищаем…" : progress > 0 ? "Продолжайте удерживать…" : "Удерживать для очистки"}</span>
    </button>
  );
}
