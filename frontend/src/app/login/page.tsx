"use client";

import { useState } from "react";
import Link from "next/link";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { motion } from "framer-motion";
import { Sparkles, ArrowRight, Lock, User, ShieldCheck } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert } from "@/components/ui/alert";
import { GradientText } from "@/components/ui/gradient-text";
import { GradientBlob } from "@/components/ui/gradient-blob";
import { useAuth } from "@/hooks/use-auth";
import { ApiError } from "@/lib/api";

const schema = z.object({
  username: z.string().min(1, "Username is required"),
  password: z.string().min(1, "Password is required"),
});

type FormValues = z.infer<typeof schema>;

export default function LoginPage() {
  const { login } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  async function onSubmit(values: FormValues) {
    setServerError(null);
    try {
      await login(values.username, values.password);
      toast.success("Welcome back");
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Could not sign in";
      setServerError(message);
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-background bg-grid px-6">
      <GradientBlob className="-top-32 left-1/4 opacity-60" />
      <GradientBlob className="bottom-0 right-1/4 opacity-40" />

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative z-10 w-full max-w-md rounded-2xl border border-surface-border bg-surface/70 p-8 shadow-2xl backdrop-blur-xl"
      >
        <Link href="/" className="flex items-center justify-center gap-2 text-lg font-semibold">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent-2">
            <Sparkles className="h-4 w-4 text-white" />
          </span>
          DataPilot
        </Link>

        <h1 className="mt-6 text-center text-3xl font-bold tracking-tight">
          Sign in to the <GradientText>console</GradientText>
        </h1>
        <p className="mt-2 text-center text-base text-muted">
          Use the credentials your administrator configured for this deployment.
        </p>

        {serverError && (
          <Alert variant="danger" className="mt-6">
            {serverError}
          </Alert>
        )}

        <form onSubmit={handleSubmit(onSubmit)} className="mt-8 space-y-5">
          <div>
            <Label htmlFor="username">Username</Label>
            <div className="relative">
              <User className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
              <Input
                id="username"
                placeholder="admin"
                className="pl-10"
                error={Boolean(errors.username)}
                {...register("username")}
              />
            </div>
            {errors.username && <p className="mt-1.5 text-xs text-danger">{errors.username.message}</p>}
          </div>

          <div>
            <Label htmlFor="password">Password</Label>
            <div className="relative">
              <Lock className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                className="pl-10"
                error={Boolean(errors.password)}
                {...register("password")}
              />
            </div>
            {errors.password && <p className="mt-1.5 text-xs text-danger">{errors.password.message}</p>}
          </div>

          <Button type="submit" className="w-full" size="lg" loading={isSubmitting}>
            Sign in <ArrowRight className="h-4 w-4" />
          </Button>
        </form>

        <div className="mt-6 flex items-start gap-3 rounded-xl border border-surface-border bg-surface-2/50 p-4">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary-2" />
          <p className="text-sm text-muted">
            DataPilot runs as a single-operator deployment — there&apos;s no public sign-up. Your
            administrator provisions the one account above via{" "}
            <code className="rounded bg-surface px-1.5 py-0.5 text-xs text-foreground">
              DATAPILOT_AUTH_USERNAME
            </code>{" "}
            /{" "}
            <code className="rounded bg-surface px-1.5 py-0.5 text-xs text-foreground">
              DATAPILOT_AUTH_PASSWORD
            </code>
            . <Link href="/" className="font-medium text-primary-2 hover:underline">
              Learn more on the landing page.
            </Link>
          </p>
        </div>
      </motion.div>
    </div>
  );
}
