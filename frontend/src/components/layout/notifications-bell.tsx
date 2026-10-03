"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Bell } from "lucide-react";
import { cn, formatRelativeTime } from "@/lib/utils";
import { getHistory, ActivityRecord, ApiError } from "@/lib/api";
import { ACTIVITY_META } from "@/lib/activity-meta";

export function NotificationsBell() {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<ActivityRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  useEffect(() => {
    if (!open || items !== null) return;
    getHistory(8)
      .then(setItems)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Could not load activity"));
  }, [open, items]);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="relative rounded-xl border border-surface-border bg-surface-2/50 p-2.5 text-muted transition-colors hover:border-primary/30 hover:text-foreground"
      >
        <Bell className="h-4 w-4" />
      </button>

      {open && (
        <div className="absolute right-0 top-full z-40 mt-2 w-80 rounded-2xl border border-surface-border bg-surface shadow-2xl">
          <div className="border-b border-surface-border px-4 py-3">
            <p className="text-sm font-semibold">Recent activity</p>
          </div>
          <div className="max-h-96 overflow-y-auto p-2">
            {error && <p className="px-3 py-6 text-center text-sm text-danger">{error}</p>}
            {!error && items === null && (
              <p className="px-3 py-6 text-center text-sm text-muted">Loading…</p>
            )}
            {!error && items !== null && items.length === 0 && (
              <p className="px-3 py-6 text-center text-sm text-muted">
                Nothing yet — ingest a dataset to get started.
              </p>
            )}
            {items?.map((item) => {
              const meta = ACTIVITY_META[item.kind];
              const Icon = meta?.icon ?? Bell;
              return (
                <div key={item.activity_id} className="flex items-start gap-3 rounded-xl px-3 py-2.5 hover:bg-white/5">
                  <span className={cn("mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/5", meta?.color)}>
                    <Icon className="h-4 w-4" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">{meta?.label ?? item.kind}</p>
                    <p className="truncate text-xs text-muted">{item.summary}</p>
                    <p className="mt-0.5 text-[11px] text-muted">{formatRelativeTime(item.created_at)}</p>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="border-t border-surface-border p-2">
            <Link
              href="/dashboard/history"
              onClick={() => setOpen(false)}
              className="block rounded-xl px-3 py-2 text-center text-sm font-medium text-primary-2 hover:bg-white/5"
            >
              View full history
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
