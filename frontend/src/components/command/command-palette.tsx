"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import {
  Search,
  LayoutDashboard,
  UploadCloud,
  ShieldCheck,
  LineChart,
  Cpu,
  ListChecks,
  Database,
  LayoutGrid,
  Target,
  History,
  Zap,
  UserCircle,
  Sun,
  Moon,
  LogOut,
  CornerDownLeft,
  Boxes,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/use-auth";
import { useTheme } from "@/components/theme/theme-provider";

interface CommandItem {
  id: string;
  label: string;
  group: "Navigate" | "Actions";
  icon: typeof Search;
  run: () => void;
  keywords?: string;
}

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();
  const { logout } = useAuth();
  const { theme, toggleTheme } = useTheme();

  const openPalette = useCallback(() => {
    setQuery("");
    setActiveIndex(0);
    setOpen(true);
    requestAnimationFrame(() => inputRef.current?.focus());
  }, []);

  useEffect(() => {
    function handleKeydown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (open) setOpen(false);
        else openPalette();
      }
    }
    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [open, openPalette]);

  const items: CommandItem[] = useMemo(
    () => [
      { id: "overview", label: "Overview", group: "Navigate", icon: LayoutDashboard, run: () => router.push("/dashboard") },
      { id: "all-in-one", label: "All in one go", group: "Navigate", icon: Zap, run: () => router.push("/dashboard/all-in-one") },
      { id: "ingest", label: "Ingest", group: "Navigate", icon: UploadCloud, run: () => router.push("/dashboard/upload") },
      { id: "quality", label: "Quality", group: "Navigate", icon: ShieldCheck, run: () => router.push("/dashboard/quality") },
      { id: "eda", label: "EDA", group: "Navigate", icon: LineChart, run: () => router.push("/dashboard/eda") },
      { id: "dashboards", label: "Dashboards", group: "Navigate", icon: LayoutGrid, run: () => router.push("/dashboard/dashboards") },
      { id: "modeling", label: "Modeling", group: "Navigate", icon: Cpu, run: () => router.push("/dashboard/modeling") },
      { id: "clustering", label: "Clustering", group: "Navigate", icon: Boxes, run: () => router.push("/dashboard/clustering") },
      { id: "predict", label: "Train & Predict", group: "Navigate", icon: Target, run: () => router.push("/dashboard/predict") },
      { id: "jobs", label: "Jobs", group: "Navigate", icon: ListChecks, run: () => router.push("/dashboard/jobs") },
      { id: "analytics", label: "Analytics", group: "Navigate", icon: Database, run: () => router.push("/dashboard/analytics") },
      { id: "history", label: "History", group: "Navigate", icon: History, run: () => router.push("/dashboard/history") },
      { id: "profile", label: "Profile", group: "Navigate", icon: UserCircle, run: () => router.push("/dashboard/profile") },
      {
        id: "theme",
        label: theme === "dark" ? "Switch to light mode" : "Switch to dark mode",
        group: "Actions",
        icon: theme === "dark" ? Sun : Moon,
        run: toggleTheme,
        keywords: "theme dark light appearance",
      },
      { id: "logout", label: "Sign out", group: "Actions", icon: LogOut, run: logout },
    ],
    [router, theme, toggleTheme, logout],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (item) =>
        item.label.toLowerCase().includes(q) || item.keywords?.toLowerCase().includes(q),
    );
  }, [items, query]);

  function select(item: CommandItem) {
    item.run();
    setOpen(false);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIndex((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const item = filtered[activeIndex];
      if (item) select(item);
    }
  }

  const navItems = filtered.filter((i) => i.group === "Navigate");
  const actionItems = filtered.filter((i) => i.group === "Actions");

  return (
    <>
      <button
        onClick={openPalette}
        className="flex items-center gap-2 rounded-xl border border-surface-border bg-surface-2/50 px-3.5 py-2 text-sm text-muted transition-colors hover:border-primary/30 hover:text-foreground"
      >
        <Search className="h-3.5 w-3.5" />
        <span className="hidden sm:inline">Search or jump to...</span>
        <kbd className="ml-2 hidden rounded border border-surface-border bg-surface px-1.5 py-0.5 font-mono text-[10px] text-muted sm:inline">
          &#8984;K
        </kbd>
      </button>

      <DialogPrimitive.Root
        open={open}
        onOpenChange={(next) => (next ? openPalette() : setOpen(false))}
      >
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm" />
          <DialogPrimitive.Content
            className="fixed left-1/2 top-[18%] z-50 w-full max-w-xl -translate-x-1/2 overflow-hidden rounded-2xl border border-surface-border bg-surface shadow-2xl"
            onKeyDown={handleKeyDown}
          >
            <DialogPrimitive.Title className="sr-only">Command palette</DialogPrimitive.Title>
            <div className="flex items-center gap-3 border-b border-surface-border px-4 py-3.5">
              <Search className="h-4 w-4 text-muted" />
              <input
                ref={inputRef}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setActiveIndex(0);
                }}
                placeholder="Type a page name or action..."
                className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted"
              />
              <kbd className="rounded border border-surface-border px-1.5 py-0.5 font-mono text-[10px] text-muted">
                esc
              </kbd>
            </div>

            <div className="max-h-80 overflow-y-auto p-2">
              {filtered.length === 0 && (
                <p className="px-3 py-6 text-center text-sm text-muted">No matches.</p>
              )}
              {navItems.length > 0 && (
                <CommandGroup
                  label="Navigate"
                  groupItems={navItems}
                  allItems={filtered}
                  activeIndex={activeIndex}
                  onSelect={select}
                  onHover={setActiveIndex}
                />
              )}
              {actionItems.length > 0 && (
                <CommandGroup
                  label="Actions"
                  groupItems={actionItems}
                  allItems={filtered}
                  activeIndex={activeIndex}
                  onSelect={select}
                  onHover={setActiveIndex}
                />
              )}
            </div>
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>
    </>
  );
}

function CommandGroup({
  label,
  groupItems,
  allItems,
  activeIndex,
  onSelect,
  onHover,
}: {
  label: string;
  groupItems: CommandItem[];
  allItems: CommandItem[];
  activeIndex: number;
  onSelect: (item: CommandItem) => void;
  onHover: (index: number) => void;
}) {
  return (
    <div className="mb-1">
      <p className="px-3 py-1.5 text-[11px] font-medium uppercase tracking-wide text-muted">{label}</p>
      {groupItems.map((item) => {
        const index = allItems.indexOf(item);
        const active = index === activeIndex;
        return (
          <button
            key={item.id}
            onMouseEnter={() => onHover(index)}
            onClick={() => onSelect(item)}
            className={cn(
              "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm transition-colors",
              active ? "bg-primary/10 text-foreground" : "text-muted hover:bg-white/5",
            )}
          >
            <item.icon className="h-4 w-4 shrink-0" />
            <span className="flex-1">{item.label}</span>
            {active && <CornerDownLeft className="h-3.5 w-3.5 shrink-0 text-muted" />}
          </button>
        );
      })}
    </div>
  );
}
