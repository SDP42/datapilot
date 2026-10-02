import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Card, CardContent } from "./card";

export function StatCard({
  icon,
  label,
  value,
  trend,
  className,
}: {
  icon?: ReactNode;
  label: string;
  value: ReactNode;
  trend?: { value: string; positive?: boolean };
  className?: string;
}) {
  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardContent className="flex flex-col gap-3 p-5">
        <div className="flex items-center justify-between">
          {icon && (
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 text-primary-2">
              {icon}
            </div>
          )}
          {trend && (
            <span
              className={cn(
                "shrink-0 rounded-full px-2 py-0.5 text-sm font-medium",
                trend.positive ? "bg-success/10 text-success" : "bg-danger/10 text-danger",
              )}
            >
              {trend.value}
            </span>
          )}
        </div>
        <div className="min-w-0">
          <p className="text-3xl font-bold tracking-tight">{value}</p>
          <p className="mt-1 break-words text-sm font-medium text-muted">{label}</p>
        </div>
      </CardContent>
    </Card>
  );
}
