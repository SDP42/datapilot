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
    <div className="min-h-screen bg-background bg-grid">
      <Sidebar />
      <main className="min-h-screen lg:pl-64">
        <div className="mx-auto max-w-6xl px-6 py-8 sm:px-8">{children}</div>
      </main>
    </div>
  );
}
