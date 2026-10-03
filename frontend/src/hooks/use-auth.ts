"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  login as apiLogin,
  register as apiRegister,
  getMe,
  RegisterInput,
  UserProfile,
} from "@/lib/api";

interface AuthState {
  username: string | null;
  profile: UserProfile | null;
  loading: boolean;
}

export function useAuth() {
  const router = useRouter();
  const [state, setState] = useState<AuthState>({ username: null, profile: null, loading: true });

  const loadProfile = useCallback(async () => {
    const me = await getMe();
    setState({ username: me.username, profile: me, loading: false });
    return me;
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function resolveSession() {
      const token = window.localStorage.getItem("dp_token");
      if (!token) {
        if (!cancelled) setState({ username: null, profile: null, loading: false });
        return;
      }
      try {
        const me = await getMe();
        if (!cancelled) setState({ username: me.username, profile: me, loading: false });
      } catch {
        window.localStorage.removeItem("dp_token");
        if (!cancelled) setState({ username: null, profile: null, loading: false });
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
      setState((s) => ({ ...s, username: result.username, loading: false }));
      await loadProfile();
      router.push("/dashboard");
    },
    [router, loadProfile],
  );

  const register = useCallback(
    async (input: RegisterInput) => {
      const result = await apiRegister(input);
      window.localStorage.setItem("dp_token", result.access_token);
      setState((s) => ({ ...s, username: result.username, loading: false }));
      await loadProfile();
      router.push("/dashboard");
    },
    [router, loadProfile],
  );

  const logout = useCallback(() => {
    window.localStorage.removeItem("dp_token");
    setState({ username: null, profile: null, loading: false });
    router.push("/login");
  }, [router]);

  return {
    ...state,
    login,
    register,
    logout,
    refreshProfile: loadProfile,
    isAuthenticated: Boolean(state.username),
  };
}
