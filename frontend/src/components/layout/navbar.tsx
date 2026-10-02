"use client";

import Link from "next/link";
import { useState } from "react";
import { motion } from "framer-motion";
import { Menu, X, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

const LINKS = [
  { href: "#features", label: "Features" },
  { href: "#pipeline", label: "Pipeline" },
  { href: "#stack", label: "Stack" },
];

export function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <motion.header
      initial={{ y: -40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="fixed inset-x-0 top-0 z-40 border-b border-surface-border/60 glass"
    >
      <nav className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent-2">
            <Sparkles className="h-4 w-4 text-white" />
          </span>
          DataPilot
        </Link>

        <div className="hidden items-center gap-8 md:flex">
          {LINKS.map((link) => (
            <a key={link.href} href={link.href} className="text-sm text-muted transition-colors hover:text-foreground">
              {link.label}
            </a>
          ))}
        </div>

        <div className="hidden items-center gap-3 md:flex">
          <Link href="/login">
            <Button variant="ghost" size="sm">
              Sign in
            </Button>
          </Link>
          <Link href="/login">
            <Button size="sm">Get started</Button>
          </Link>
        </div>

        <button className="md:hidden" onClick={() => setOpen(!open)}>
          {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </button>
      </nav>

      {open && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          className="overflow-hidden border-t border-surface-border px-6 pb-6 md:hidden"
        >
          <div className="flex flex-col gap-4 pt-4">
            {LINKS.map((link) => (
              <a key={link.href} href={link.href} className="text-sm text-muted" onClick={() => setOpen(false)}>
                {link.label}
              </a>
            ))}
            <Link href="/login" onClick={() => setOpen(false)}>
              <Button className="w-full">Get started</Button>
            </Link>
          </div>
        </motion.div>
      )}
    </motion.header>
  );
}
