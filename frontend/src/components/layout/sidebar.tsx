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
  LayoutGrid,
  Target,
  History,
  Zap,
  Sun,
  Moon,
  Flame,
  UserCircle,
  Boxes,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/use-auth";
import { Avatar } from "@/components/ui/avatar";
import { useTheme } from "@/components/theme/theme-provider";

const NAV = [
  { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { href: "/dashboard/all-in-one", label: "All in one go", icon: Zap },
  { href: "/dashboard/upload", label: "Ingest", icon: UploadCloud },
  { href: "/dashboard/quality", label: "Quality", icon: ShieldCheck },
  { href: "/dashboard/eda", label: "EDA", icon: LineChart },
  { href: "/dashboard/dashboards", label: "Dashboards", icon: LayoutGrid },
  { href: "/dashboard/modeling", label: "Modeling", icon: Cpu },
  { href: "/dashboard/clustering", label: "Clustering", icon: Boxes },
  { href: "/dashboard/predict", label: "Train & Predict", icon: Target },
  { href: "/dashboard/jobs", label: "Jobs", icon: ListChecks },
  { href: "/dashboard/analytics", label: "Analytics", icon: Database },
  { href: "/dashboard/history", label: "History", icon: History },
  { href: "/dashboard/profile", label: "Profile", icon: UserCircle },
];

export function Sidebar() {
  const pathname = usePathname();
  const { username, profile, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();

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

      <div className="flex items-center justify-between gap-1 border-t border-surface-border px-4 py-2">
        <button
          onClick={toggleTheme}
          className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-muted hover:bg-white/5 hover:text-foreground"
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
          {theme === "dark" ? "Light" : "Dark"}
        </button>
        {profile && (
          <span className="flex items-center gap-1 text-xs text-muted">
            <Flame className="h-3.5 w-3.5 text-warning" /> {profile.current_streak}
          </span>
        )}
      </div>

      <Link
        href="/dashboard/profile"
        className="flex items-center gap-3 border-t border-surface-border p-4 hover:bg-white/5"
      >
        <Avatar name={profile?.full_name || username || "user"} size={34} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{username ?? "Guest"}</p>
          <p className="text-xs text-muted">
            {profile ? `Level ${profile.level} · ${profile.xp} XP` : "Signed in"}
          </p>
        </div>
        <button
          onClick={(e) => {
            e.preventDefault();
            logout();
          }}
          className="rounded-lg p-2 text-muted hover:bg-white/5 hover:text-danger"
        >
          <LogOut className="h-4 w-4" />
        </button>
      </Link>
    </aside>
  );
}
