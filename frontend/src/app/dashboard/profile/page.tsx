"use client";

import { Flame, Trophy, Award, Sparkles, GraduationCap, Target } from "lucide-react";

import { PageHeader } from "@/components/layout/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar } from "@/components/ui/avatar";
import { Spinner } from "@/components/ui/spinner";
import { useAuth } from "@/hooks/use-auth";

const XP_PER_LEVEL = 100;

const GOAL_LABELS: Record<string, string> = {
  learn_data_science: "Learn data science",
  analyze_business_data: "Analyze business data",
  build_ml_models: "Build ML models",
  research: "Research",
  explore_the_platform: "Exploring the platform",
  other: "Something else",
};

export default function ProfilePage() {
  const { profile, loading } = useAuth();

  if (loading || !profile) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <Spinner size={28} />
      </div>
    );
  }

  const xpIntoLevel = profile.xp % XP_PER_LEVEL;
  const xpProgress = Math.round((xpIntoLevel / XP_PER_LEVEL) * 100);

  return (
    <div>
      <PageHeader
        title="Your profile"
        description="Account details, onboarding answers, and your DataPilot activity progress."
      />

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardContent className="flex flex-col items-center py-8 text-center">
            <Avatar name={profile.full_name || profile.username} size={72} />
            <h2 className="mt-4 text-xl font-semibold">{profile.full_name || profile.username}</h2>
            <p className="text-sm text-muted">@{profile.username}</p>

            <div className="mt-5 flex flex-wrap items-center justify-center gap-2">
              <Badge variant="primary">
                <GraduationCap className="h-3 w-3" /> {profile.experience_level}
              </Badge>
              <Badge variant="accent">
                <Target className="h-3 w-3" /> {GOAL_LABELS[profile.primary_goal] ?? profile.primary_goal}
              </Badge>
            </div>

            {profile.role && <p className="mt-3 text-xs text-muted">{profile.role}</p>}
          </CardContent>
        </Card>

        <div className="space-y-6 lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary-2" /> Level {profile.level}
              </CardTitle>
              <CardDescription>
                {profile.xp} XP total — {XP_PER_LEVEL - xpIntoLevel} XP to level {profile.level + 1}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-3 w-full overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-primary to-accent-2 transition-all"
                  style={{ width: `${xpProgress}%` }}
                />
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4 sm:grid-cols-2">
            <Card>
              <CardContent className="flex items-center gap-4 py-6">
                <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-warning/10 text-warning">
                  <Flame className="h-5 w-5" />
                </span>
                <div>
                  <p className="text-2xl font-bold">{profile.current_streak}</p>
                  <p className="text-xs text-muted">day streak</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="flex items-center gap-4 py-6">
                <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary-2">
                  <Trophy className="h-5 w-5" />
                </span>
                <div>
                  <p className="text-2xl font-bold">{profile.longest_streak}</p>
                  <p className="text-xs text-muted">longest streak</p>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Award className="h-4 w-4 text-accent-2" /> Badges
              </CardTitle>
              <CardDescription>
                Earned automatically from your real activity — never awarded for free.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {profile.badges.length === 0 ? (
                <p className="text-sm text-muted">
                  No badges yet — ingest a dataset or run an analysis to earn your first one.
                </p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {profile.badges.map((badge) => (
                    <Badge key={badge} variant="success">
                      <Award className="h-3 w-3" /> {badge}
                    </Badge>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
