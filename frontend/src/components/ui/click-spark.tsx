"use client";

import { ReactNode, useState, MouseEvent } from "react";

interface Spark {
  id: number;
  x: number;
  y: number;
}

export function ClickSpark({ children, className }: { children: ReactNode; className?: string }) {
  const [sparks, setSparks] = useState<Spark[]>([]);

  function handleClick(e: MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const id = Date.now();
    setSparks((prev) => [
      ...prev,
      { id, x: e.clientX - rect.left, y: e.clientY - rect.top },
    ]);
    setTimeout(() => setSparks((prev) => prev.filter((s) => s.id !== id)), 650);
  }

  return (
    <div className={`relative overflow-hidden ${className ?? ""}`} onClick={handleClick}>
      {children}
      {sparks.map((s) => (
        <span
          key={s.id}
          className="pointer-events-none absolute h-3 w-3 rounded-full bg-accent/70 animate-spark"
          style={{ left: s.x - 6, top: s.y - 6 }}
        />
      ))}
    </div>
  );
}
