"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { motion, AnimatePresence } from "framer-motion";
import {
  Lock,
  User,
  ArrowRight,
  ArrowLeft,
  GraduationCap,
  Sparkles as SparkIcon,
  Briefcase,
  Target,
  Microscope,
  Compass,
  MoreHorizontal,
} from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/use-auth";
import { ApiError, ExperienceLevel, PrimaryGoal } from "@/lib/api";

const accountSchema = z
  .object({
    fullName: z.string().optional(),
    username: z.string().min(3, "At least 3 characters"),
    password: z.string().min(8, "At least 8 characters"),
    confirmPassword: z.string().min(1, "Please confirm your password"),
  })
  .refine((v) => v.password === v.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
  });

type AccountValues = z.infer<typeof accountSchema>;

const EXPERIENCE_OPTIONS: { value: ExperienceLevel; label: string; description: string }[] = [
  { value: "beginner", label: "Beginner", description: "New to data science" },
  { value: "intermediate", label: "Intermediate", description: "Comfortable with the basics" },
  { value: "advanced", label: "Advanced", description: "Experienced practitioner" },
];

const GOAL_OPTIONS: { value: PrimaryGoal; label: string; icon: typeof Target }[] = [
  { value: "learn_data_science", label: "Learn data science", icon: GraduationCap },
  { value: "analyze_business_data", label: "Analyze business data", icon: Briefcase },
  { value: "build_ml_models", label: "Build ML models", icon: Target },
  { value: "research", label: "Research", icon: Microscope },
  { value: "explore_the_platform", label: "Just exploring", icon: Compass },
  { value: "other", label: "Something else", icon: MoreHorizontal },
];

export function RegisterForm() {
  const { register: registerAccount } = useAuth();
  const [step, setStep] = useState<1 | 2>(1);
  const [serverError, setServerError] = useState<string | null>(null);
  const [experienceLevel, setExperienceLevel] = useState<ExperienceLevel | null>(null);
  const [primaryGoal, setPrimaryGoal] = useState<PrimaryGoal | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const {
    register,
    handleSubmit,
    trigger,
    formState: { errors },
  } = useForm<AccountValues>({ resolver: zodResolver(accountSchema) });

  async function goToStep2() {
    const valid = await trigger(["username", "password", "confirmPassword"]);
    if (valid) setStep(2);
  }

  async function onSubmit(values: AccountValues) {
    if (!experienceLevel || !primaryGoal) return;
    setServerError(null);
    setSubmitting(true);
    try {
      await registerAccount({
        username: values.username,
        password: values.password,
        confirmPassword: values.confirmPassword,
        fullName: values.fullName,
        experienceLevel,
        primaryGoal,
      });
      toast.success("Account created — welcome to DataPilot");
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Could not create account";
      setServerError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      {serverError && (
        <Alert variant="danger" className="mt-6">
          {serverError}
        </Alert>
      )}

      <form onSubmit={handleSubmit(onSubmit)} className="mt-6">
        <AnimatePresence mode="wait">
          {step === 1 ? (
            <motion.div
              key="step1"
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -12 }}
              transition={{ duration: 0.2 }}
              className="space-y-5"
            >
              <div>
                <Label htmlFor="fullName">Your name (optional)</Label>
                <div className="relative">
                  <SparkIcon className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
                  <Input
                    id="fullName"
                    placeholder="Ada Lovelace"
                    className="pl-10"
                    {...register("fullName")}
                  />
                </div>
              </div>

              <div>
                <Label htmlFor="reg-username">Username</Label>
                <div className="relative">
                  <User className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
                  <Input
                    id="reg-username"
                    placeholder="pick a username"
                    className="pl-10"
                    error={Boolean(errors.username)}
                    {...register("username")}
                  />
                </div>
                {errors.username && (
                  <p className="mt-1.5 text-xs text-danger">{errors.username.message}</p>
                )}
              </div>

              <div>
                <Label htmlFor="reg-password">Password</Label>
                <div className="relative">
                  <Lock className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
                  <Input
                    id="reg-password"
                    type="password"
                    placeholder="at least 8 characters"
                    className="pl-10"
                    error={Boolean(errors.password)}
                    {...register("password")}
                  />
                </div>
                {errors.password && (
                  <p className="mt-1.5 text-xs text-danger">{errors.password.message}</p>
                )}
              </div>

              <div>
                <Label htmlFor="reg-confirm">Confirm password</Label>
                <div className="relative">
                  <Lock className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
                  <Input
                    id="reg-confirm"
                    type="password"
                    placeholder="type it again"
                    className="pl-10"
                    error={Boolean(errors.confirmPassword)}
                    {...register("confirmPassword")}
                  />
                </div>
                {errors.confirmPassword && (
                  <p className="mt-1.5 text-xs text-danger">{errors.confirmPassword.message}</p>
                )}
              </div>

              <Button type="button" className="w-full" size="lg" onClick={goToStep2}>
                Continue <ArrowRight className="h-4 w-4" />
              </Button>
            </motion.div>
          ) : (
            <motion.div
              key="step2"
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -12 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              <div>
                <Label>How would you describe your data science experience?</Label>
                <div className="mt-2 grid grid-cols-1 gap-2">
                  {EXPERIENCE_OPTIONS.map((opt) => (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setExperienceLevel(opt.value)}
                      className={cn(
                        "rounded-xl border px-4 py-3 text-left transition-colors",
                        experienceLevel === opt.value
                          ? "border-primary/40 bg-primary/10"
                          : "border-surface-border bg-surface-2/40 hover:bg-white/5",
                      )}
                    >
                      <p className="text-sm font-medium">{opt.label}</p>
                      <p className="text-xs text-muted">{opt.description}</p>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <Label>What brings you to DataPilot?</Label>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  {GOAL_OPTIONS.map((opt) => {
                    const Icon = opt.icon;
                    return (
                      <button
                        key={opt.value}
                        type="button"
                        onClick={() => setPrimaryGoal(opt.value)}
                        className={cn(
                          "flex flex-col items-start gap-2 rounded-xl border px-3.5 py-3 text-left transition-colors",
                          primaryGoal === opt.value
                            ? "border-primary/40 bg-primary/10"
                            : "border-surface-border bg-surface-2/40 hover:bg-white/5",
                        )}
                      >
                        <Icon className="h-4 w-4 text-primary-2" />
                        <span className="text-xs font-medium leading-tight">{opt.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="flex gap-3">
                <Button type="button" variant="secondary" onClick={() => setStep(1)}>
                  <ArrowLeft className="h-4 w-4" /> Back
                </Button>
                <Button
                  type="submit"
                  className="flex-1"
                  size="lg"
                  loading={submitting}
                  disabled={!experienceLevel || !primaryGoal}
                >
                  Create account <ArrowRight className="h-4 w-4" />
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </form>
    </div>
  );
}
