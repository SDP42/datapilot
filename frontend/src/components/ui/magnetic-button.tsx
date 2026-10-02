"use client";

import { useRef, MouseEvent, ReactNode } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";
import { cn } from "@/lib/utils";

export function MagneticButton({
  children,
  className,
  onClick,
  strength = 0.4,
}: {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  strength?: number;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const springX = useSpring(x, { stiffness: 200, damping: 15 });
  const springY = useSpring(y, { stiffness: 200, damping: 15 });

  function handleMove(e: MouseEvent<HTMLButtonElement>) {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    x.set((e.clientX - rect.left - rect.width / 2) * strength);
    y.set((e.clientY - rect.top - rect.height / 2) * strength);
  }

  function handleLeave() {
    x.set(0);
    y.set(0);
  }

  return (
    <motion.button
      ref={ref}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
      onClick={onClick}
      style={{ x: springX, y: springY }}
      className={cn(
        "glow-border relative inline-flex items-center justify-center rounded-xl bg-gradient-to-r from-primary to-primary-2 px-7 py-3.5 text-lg font-medium text-white shadow-xl shadow-primary/30",
        className,
      )}
    >
      {children}
    </motion.button>
  );
}
