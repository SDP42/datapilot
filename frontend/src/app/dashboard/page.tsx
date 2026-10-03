"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import {
  Database,
  Cpu,
  ListChecks,
  ShieldCheck,
  ArrowRight,
  Flame,
  Sparkles,
  Trophy,
} from "lucide-react";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { PageHeader } from "@/components/layout/page-header";
import { StatCard } from "@/components/ui/stat-card";
import { SpotlightCard } from "@/components/ui/spotlight-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { useAuth } from "@/hooks/use-auth";
import { getHistory, ActivityRecord, ApiError } from "@/lib/api";
import { ACTIVITY_META } from "@/lib/activity-meta";
import { formatRelativeTime } from "@/lib/utils";

const DashboardOrb = dynamic(
  () => import("@/components/ui/dashboard-orb").then((m) => m.DashboardOrb),
  { ssr: false },
);

const QUICK_LINKS = [
  {
    href: "/dashboard/upload",
    icon: Database,
    title: "Ingest a dataset",
    desc: "Upload a CSV or Excel file and get an instant structured profile.",
  },
  {
    href: "/dashboard/quality",
    icon: ShieldCheck,
    title: "Run quality analysis",
    desc: "Missing values, duplicates, outliers, skew, imbalance.",
  },
  {
    href: "/dashboard/modeling",
    icon: Cpu,
    title: "Run the modeling pipeline",
    desc: "Readiness, candidates, training, evaluation, selection.",
  },
  {
    href: "/dashboard/jobs",
    icon: ListChecks,
    title: "Track background jobs",
    desc: "Submit a long-running run and poll it to completion.",
  },
];

function activityByDay(items: ActivityRecord[]): { day: string; runs: number }[] {
  const buckets = new Map<string, number>();
  const now = new Date();
  for (let i = 13; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    buckets.set(key, 0);
  }
  for (const item of items) {
    const key = item.created_at.slice(0, 10);
    if (buckets.has(key)) buckets.set(key, (buckets.get(key) ?? 0) + 1);
  }
  return [...buckets.entries()].map(([day, runs]) => ({
    day: new Date(day).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
    runs,
  }));
}

export default function DashboardOverviewPage() {
  const { username, profile } = useAuth();
  const [history, setHistory] = useState<ActivityRecord[] | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHistory(100)
      .then((items) => {
        if (!cancelled) setHistory(items);
      })
      .catch((err) => {
        if (!cancelled) {
          setHistoryError(err instanceof ApiError ? err.message : "Could not load activity");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const chartData = useMemo(() => activityByDay(history ?? []), [history]);
  const recent = history?.slice(0, 6) ?? [];
  const runsToday = useMemo(() => {
    if (!history) return 0;
    const today = new Date().toISOString().slice(0, 10);
    return history.filter((h) => h.created_at.slice(0, 10) === today).length;
  }, [history]);

  const xpIntoLevel = profile ? profile.xp % 100 : 0;

  return (
    <div>
      <div className="relative overflow-hidden">
        <div className="pointer-events-none absolute -right-6 -top-24 -z-10 h-48 w-48 opacity-40 sm:h-60 sm:w-60">
          <DashboardOrb className="h-full w-full" />
        </div>
        <PageHeader
          title={`Welcome back${username ? `, ${username}` : ""}`}
          description="Pick a phase of the pipeline to run against a dataset."
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          icon={<Database className="h-5 w-5" />}
          label="Total runs"
          value={history ? history.length : <Spinner size={16} />}
        />
        <StatCard
          icon={<Cpu className="h-5 w-5" />}
          label="Runs today"
          value={history ? runsToday : <Spinner size={16} />}
        />
        <StatCard
          icon={<Sparkles className="h-5 w-5" />}
          label="Level"
          value={profile ? profile.level : <Spinner size={16} />}
        />
        <StatCard
          icon={<Flame className="h-5 w-5" />}
          label="Day streak"
          value={profile ? profile.current_streak : <Spinner size={16} />}
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Activity, last 14 days</CardTitle>
          </CardHeader>
          <CardContent className="h-56">
            {historyError ? (
              <p className="flex h-full items-center justify-center text-sm text-danger">
                {historyError}
              </p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="activityFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="var(--primary)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="day"
                    tick={{ fontSize: 10, fill: "var(--muted)" }}
                    interval={2}
                  />
                  <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: "var(--muted)" }} width={24} />
                  <Tooltip
                    contentStyle={{
                      background: "var(--surface-2)",
                      border: "1px solid var(--surface-border)",
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="runs"
                    stroke="var(--primary)"
                    fill="url(#activityFill)"
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Trophy className="h-4 w-4 text-warning" /> Your progress
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {profile ? (
              <>
                <div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">Level {profile.level}</span>
                    <span className="text-muted">{profile.xp} XP</span>
                  </div>
                  <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-surface-2">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-primary to-accent-2"
                      style={{ width: `${xpIntoLevel}%` }}
                    />
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {profile.badges.slice(0, 3).map((badge) => (
                    <Badge key={badge} variant="success">
                      {badge}
                    </Badge>
                  ))}
                  {profile.badges.length === 0 && (
                    <p className="text-xs text-muted">No badges yet — go run something.</p>
                  )}
                </div>
                <Link href="/dashboard/profile">
                  <Button variant="secondary" size="sm" className="w-full">
                    View full profile <ArrowRight className="h-3.5 w-3.5" />
                  </Button>
                </Link>
              </>
            ) : (
              <div className="flex justify-center py-6">
                <Spinner size={20} />
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <h2 className="mb-3 text-sm font-semibold text-muted">Quick start</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {QUICK_LINKS.map((link) => (
              <SpotlightCard key={link.href}>
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 text-primary-2">
                  <link.icon className="h-5 w-5" />
                </div>
                <h3 className="mt-4 text-lg font-semibold">{link.title}</h3>
                <p className="mt-1.5 text-base text-muted">{link.desc}</p>
                <Link href={link.href} className="mt-4 inline-block">
                  <Button variant="secondary" size="sm">
                    Open <ArrowRight className="h-3.5 w-3.5" />
                  </Button>
                </Link>
              </SpotlightCard>
            ))}
          </div>
        </div>

        <div>
          <h2 className="mb-3 text-sm font-semibold text-muted">Recent activity</h2>
          <Card>
            <CardContent className="space-y-1 p-2">
              {recent.length === 0 && (
                <p className="px-3 py-6 text-center text-sm text-muted">
                  Nothing yet — ingest a dataset to get started.
                </p>
              )}
              {recent.map((item) => {
                const meta = ACTIVITY_META[item.kind];
                const Icon = meta?.icon ?? Database;
                return (
                  <div key={item.activity_id} className="flex items-start gap-3 rounded-xl px-2 py-2.5">
                    <span className={`mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white/5 ${meta?.color ?? ""}`}>
                      <Icon className="h-4 w-4" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{meta?.label ?? item.kind}</p>
                      <p className="mt-0.5 text-xs text-muted">{formatRelativeTime(item.created_at)}</p>
                    </div>
                  </div>
                );
              })}
              {recent.length > 0 && (
                <Link
                  href="/dashboard/history"
                  className="block rounded-xl px-3 py-2.5 text-center text-sm font-medium text-primary-2 hover:bg-white/5"
                >
                  View full history
                </Link>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
