"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { login as apiLogin, getMe } from "@/lib/api";

interface AuthState {
  username: string | null;
  loading: boolean;
}

export function useAuth() {
  const router = useRouter();
  const [state, setState] = useState<AuthState>({ username: null, loading: true });

  useEffect(() => {
    let cancelled = false;

    async function resolveSession() {
      const token = window.localStorage.getItem("dp_token");
      if (!token) {
        if (!cancelled) setState({ username: null, loading: false });
        return;
      }
      try {
        const me = await getMe();
        if (!cancelled) setState({ username: me.username, loading: false });
      } catch {
        window.localStorage.removeItem("dp_token");
        if (!cancelled) setState({ username: null, loading: false });
      }
    }

    void resolveSession();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(
    async (username: string, password: string) => {
      const result = await apiLogin(username, password);
      window.localStorage.setItem("dp_token", result.access_token);
      setState({ username: result.username, loading: false });
      router.push("/dashboard");
    },
    [router],
  );

  const logout = useCallback(() => {
    window.localStorage.removeItem("dp_token");
    setState({ username: null, loading: false });
    router.push("/login");
  }, [router]);

  return { ...state, login, logout, isAuthenticated: Boolean(state.username) };
}
