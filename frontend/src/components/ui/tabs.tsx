"use client";

import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cn } from "@/lib/utils";

export const Tabs = TabsPrimitive.Root;

export function TabsList({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      className={cn(
        "inline-flex items-center gap-1 rounded-xl border border-surface-border bg-surface-2/60 p-1",
        className,
      )}
      {...props}
    />
  );
}

export function TabsTrigger({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        "rounded-lg px-4 py-2 text-sm font-medium text-muted transition-all",
        "data-[state=active]:bg-gradient-to-r data-[state=active]:from-primary data-[state=active]:to-primary-2 data-[state=active]:text-white data-[state=active]:shadow-md data-[state=active]:shadow-primary/30",
        "hover:text-foreground",
        className,
      )}
      {...props}
    />
  );
}

export function TabsContent({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Content>) {
  return (
    <TabsPrimitive.Content className={cn("mt-4 animate-fade-up", className)} {...props} />
  );
}
