"use client";

import { CommandPalette } from "@/components/command/command-palette";
import { NotificationsBell } from "@/components/layout/notifications-bell";

export function TopBar() {
  return (
    <div className="sticky top-0 z-20 flex items-center justify-end gap-3 border-b border-surface-border bg-surface/60 px-6 py-3 backdrop-blur-xl sm:px-8">
      <CommandPalette />
      <NotificationsBell />
    </div>
  );
}
