"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  UploadCloud,
  ShieldCheck,
  LineChart,
  Cpu,
  ListChecks,
  Database,
  LogOut,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/use-auth";
import { Avatar } from "@/components/ui/avatar";

const NAV = [
  { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { href: "/dashboard/upload", label: "Ingest", icon: UploadCloud },
  { href: "/dashboard/quality", label: "Quality", icon: ShieldCheck },
  { href: "/dashboard/eda", label: "EDA", icon: LineChart },
  { href: "/dashboard/modeling", label: "Modeling", icon: Cpu },
  { href: "/dashboard/jobs", label: "Jobs", icon: ListChecks },
  { href: "/dashboard/analytics", label: "Analytics", icon: Database },
];

export function Sidebar() {
  const pathname = usePathname();
  const { username, logout } = useAuth();

  return (
    <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col border-r border-surface-border bg-surface/60 backdrop-blur-xl lg:flex">
      <div className="flex h-16 items-center gap-2 border-b border-surface-border px-6 font-semibold">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent-2">
          <Sparkles className="h-4 w-4 text-white" />
        </span>
        DataPilot
      </div>

      <nav className="flex-1 space-y-1 px-3 py-6">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-xl px-3.5 py-2.5 text-sm font-medium transition-colors",
                active
                  ? "bg-gradient-to-r from-primary/15 to-accent/10 text-foreground shadow-inner shadow-primary/10 border border-primary/20"
                  : "text-muted hover:bg-white/5 hover:text-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="flex items-center gap-3 border-t border-surface-border p-4">
        <Avatar name={username ?? "user"} size={34} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{username ?? "Guest"}</p>
          <p className="text-xs text-muted">Signed in</p>
        </div>
        <button onClick={logout} className="rounded-lg p-2 text-muted hover:bg-white/5 hover:text-danger">
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </aside>
  );
}
