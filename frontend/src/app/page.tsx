"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Boxes,
  Brain,
  Database,
  GitBranch,
  Lock,
  ShieldCheck,
  Sparkles,
  Workflow,
} from "lucide-react";

import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";
import { Button } from "@/components/ui/button";
import { GradientText } from "@/components/ui/gradient-text";
import { TextReveal } from "@/components/ui/text-reveal";
import { ShinyText } from "@/components/ui/shiny-text";
import { SpotlightCard } from "@/components/ui/spotlight-card";
import { TiltCard } from "@/components/ui/tilt-card";
import { MagneticButton } from "@/components/ui/magnetic-button";
import { AnimatedCounter } from "@/components/ui/animated-counter";
import { Marquee } from "@/components/ui/marquee";
import { GradientBlob } from "@/components/ui/gradient-blob";
import { ClickSpark } from "@/components/ui/click-spark";
import { Badge } from "@/components/ui/badge";

const Hero3D = dynamic(() => import("@/components/ui/hero-3d").then((m) => m.Hero3D), {
  ssr: false,
});

const FEATURES = [
  {
    icon: Database,
    title: "Deterministic data engine",
    desc: "Ingestion, profiling, quality analysis, cleaning, lineage, and EDA — every result is reproducible and explainable.",
  },
  {
    icon: Brain,
    title: "Classical + deep modeling",
    desc: "scikit-learn baselines and PyTorch architectures (MLP, CNN, LSTM, Transformer) ranked by a fixed, task-appropriate metric.",
  },
  {
    icon: ShieldCheck,
    title: "Explainability built in",
    desc: "Permutation importance, SHAP, and partial dependence — always against an already-fitted model, never fabricated.",
  },
  {
    icon: GitBranch,
    title: "Experiment tracking",
    desc: "Every run captured with environment, seed, and lineage — compared deterministically, never guessed.",
  },
  {
    icon: Workflow,
    title: "Autonomous agent",
    desc: "An LLM plans, a deterministic executor runs the tool layer, a critic decides — every step fully traced.",
  },
  {
    icon: Lock,
    title: "AI proposes, code decides",
    desc: "LLM recommendations are only ever executed after translation into a typed, validated, deterministic call.",
  },
];

const STACK = [
  "Python",
  "FastAPI",
  "PyTorch",
  "scikit-learn",
  "SQLAlchemy",
  "DuckDB",
  "Pandas",
  "Pydantic",
  "Next.js",
  "TypeScript",
  "Three.js",
  "Framer Motion",
];

const PIPELINE = [
  { phase: "01", title: "Ingest & profile", detail: "CSV in, a structured dataset profile out." },
  { phase: "02", title: "Clean & validate", detail: "Quality findings, approved cleaning, full lineage." },
  { phase: "03", title: "Understand & engineer", detail: "Target, task type, metrics, and feature recommendations." },
  { phase: "04", title: "Model & explain", detail: "Baselines, deep learning, selection, and explanations." },
  { phase: "05", title: "Track & automate", detail: "Experiments recorded, compared, and an agent that acts on them." },
];

export default function LandingPage() {
  return (
    <div className="relative flex min-h-screen flex-col overflow-x-hidden bg-background bg-grid">
      <div className="noise-overlay" />
      <Navbar />

      {/* Hero */}
      <section className="relative mx-auto flex w-full max-w-7xl flex-1 flex-col items-center px-6 pb-24 pt-40 text-center">
        <GradientBlob className="-top-40 left-1/2 -translate-x-1/2 opacity-70" />

        <div className="absolute inset-x-0 top-20 -z-0 h-[320px] opacity-50 sm:h-[420px] sm:opacity-65 lg:h-[560px] lg:opacity-80">
          <Hero3D className="h-full w-full scale-75 sm:scale-90 lg:scale-100" />
        </div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="relative z-10"
        >
          <Badge variant="primary" className="mb-6">
            <Sparkles className="h-3 w-3" />
            13 engine phases, end to end
          </Badge>
        </motion.div>

        <h1 className="relative z-10 max-w-4xl text-5xl font-bold leading-[1.1] tracking-tight sm:text-6xl md:text-7xl">
          <TextReveal text="Autonomous, explainable" />
          <br />
          <GradientText className="inline-block">
            <TextReveal text="data science — fully traced." delay={0.35} />
          </GradientText>
        </h1>

        <p className="relative z-10 mt-6 max-w-2xl text-balance text-xl text-muted">
          Upload a dataset and DataPilot ingests, profiles, cleans, explores, models, explains,
          and tracks every step deterministically — with an autonomous agent layered on top that
          proposes, never silently executes.
        </p>

        <div className="relative z-10 mt-10 flex flex-col items-center gap-4 sm:flex-row">
          <Link href="/login">
            <MagneticButton>
              Launch the console <ArrowRight className="h-4 w-4" />
            </MagneticButton>
          </Link>
          <a href="#pipeline">
            <Button variant="secondary" size="lg">
              See the pipeline
            </Button>
          </a>
        </div>

        <div className="relative z-10 mt-20 grid w-full max-w-3xl grid-cols-2 gap-8 sm:grid-cols-4">
          {[
            { value: 13, suffix: "", label: "Engine phases" },
            { value: 4, suffix: "", label: "DL architectures" },
            { value: 7, suffix: "", label: "Agent tools" },
            { value: 100, suffix: "%", label: "Traced runs" },
          ].map((stat) => (
            <div key={stat.label}>
              <AnimatedCounter
                value={stat.value}
                suffix={stat.suffix}
                className="text-3xl font-bold tracking-tight"
              />
              <p className="mt-1 text-xs text-muted">{stat.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Marquee */}
      <section className="border-y border-surface-border bg-surface/40 py-6">
        <Marquee items={STACK} />
      </section>

      {/* Features */}
      <section id="features" className="mx-auto w-full max-w-7xl px-6 py-28">
        <div className="mx-auto max-w-2xl text-center">
          <ShinyText className="text-sm font-semibold uppercase tracking-widest">
            What&apos;s inside
          </ShinyText>
          <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
            Every phase, <GradientText>one coherent platform</GradientText>
          </h2>
        </div>

        <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.45, delay: i * 0.06 }}
            >
              <SpotlightCard className="h-full">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 text-primary-2">
                  <f.icon className="h-5 w-5" />
                </div>
                <h3 className="mt-4 text-lg font-semibold">{f.title}</h3>
                <p className="mt-2 text-base text-muted">{f.desc}</p>
              </SpotlightCard>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Pipeline */}
      <section id="pipeline" className="relative mx-auto w-full max-w-7xl px-6 py-28">
        <GradientBlob className="right-0 top-20 opacity-40" />
        <div className="mx-auto max-w-2xl text-center">
          <ShinyText className="text-sm font-semibold uppercase tracking-widest">
            How it flows
          </ShinyText>
          <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">
            From raw CSV to a <GradientText>recommended model</GradientText>
          </h2>
        </div>

        <div className="relative mt-16 grid gap-6 lg:grid-cols-5">
          {PIPELINE.map((step, i) => (
            <motion.div
              key={step.phase}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.4, delay: i * 0.08 }}
              className="relative rounded-2xl border border-surface-border bg-surface/50 p-5"
            >
              <span className="text-sm font-mono text-primary-2">{step.phase}</span>
              <h3 className="mt-2 text-base font-semibold">{step.title}</h3>
              <p className="mt-1.5 text-sm text-muted">{step.detail}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="mx-auto w-full max-w-5xl px-6 pb-28">
        <ClickSpark>
          <TiltCard className="flex flex-col items-center gap-6 bg-gradient-to-br from-surface via-surface to-primary/10 px-10 py-16 text-center">
            <Boxes className="h-10 w-10 text-primary-2" />
            <h2 className="max-w-lg text-3xl font-bold tracking-tight">
              Ready to see <GradientText>deterministic AI</GradientText> in action?
            </h2>
            <p className="max-w-md text-sm text-muted">
              Sign in to the console, upload a dataset, and watch every phase run — quality,
              EDA, modeling, and explanations — in minutes.
            </p>
            <Link href="/login">
              <Button size="lg">
                Get started <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </TiltCard>
        </ClickSpark>
      </section>

      <Footer />
    </div>
  );
}
