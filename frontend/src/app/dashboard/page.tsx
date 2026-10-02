"use client";

import Link from "next/link";
import { Database, Cpu, ListChecks, ShieldCheck, ArrowRight } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { StatCard } from "@/components/ui/stat-card";
import { SpotlightCard } from "@/components/ui/spotlight-card";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";

const QUICK_LINKS = [
  {
    href: "/dashboard/upload",
    icon: Database,
    title: "Ingest a dataset",
    desc: "Upload a CSV and get an instant structured profile.",
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

export default function DashboardOverviewPage() {
  const { username } = useAuth();

  return (
    <div>
      <PageHeader
        title={`Welcome back${username ? `, ${username}` : ""}`}
        description="Pick a phase of the pipeline to run against a dataset."
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard icon={<Database className="h-5 w-5" />} label="Engine phases" value="13" />
        <StatCard icon={<Cpu className="h-5 w-5" />} label="DL architectures" value="4" />
        <StatCard icon={<ShieldCheck className="h-5 w-5" />} label="Explainability methods" value="3" />
        <StatCard icon={<ListChecks className="h-5 w-5" />} label="Agent tools" value="7" />
      </div>

      <div className="mt-10 grid gap-4 sm:grid-cols-2">
        {QUICK_LINKS.map((link) => (
          <SpotlightCard key={link.href}>
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 text-primary-2">
              <link.icon className="h-5 w-5" />
            </div>
            <h3 className="mt-4 text-base font-semibold">{link.title}</h3>
            <p className="mt-1.5 text-sm text-muted">{link.desc}</p>
            <Link href={link.href} className="mt-4 inline-block">
              <Button variant="secondary" size="sm">
                Open <ArrowRight className="h-3.5 w-3.5" />
              </Button>
            </Link>
          </SpotlightCard>
        ))}
      </div>
    </div>
  );
}
