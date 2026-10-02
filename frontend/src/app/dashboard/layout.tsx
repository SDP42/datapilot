"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/layout/sidebar";
import { Spinner } from "@/components/ui/spinner";
import { useAuth } from "@/hooks/use-auth";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { loading, isAuthenticated } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !isAuthenticated) router.replace("/login");
  }, [loading, isAuthenticated, router]);

  if (loading || !isAuthenticated) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Spinner size={28} />
      </div>
    );
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-background bg-grid">
      <div className="aurora-field opacity-40" />
      <div className="noise-overlay" />
      <Sidebar />
      <main className="relative z-10 min-h-screen lg:pl-64">
        <div className="mx-auto max-w-6xl px-6 py-8 sm:px-8">{children}</div>
      </main>
    </div>
  );
}
