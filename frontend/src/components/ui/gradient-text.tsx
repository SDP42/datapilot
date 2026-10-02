import { cn } from "@/lib/utils";

export function GradientText({
  children,
  className,
  animate = true,
}: {
  children: React.ReactNode;
  className?: string;
  animate?: boolean;
}) {
  return (
    <span className={cn("text-gradient", animate && "animate-gradient-shift", className)}>
      {children}
    </span>
  );
}
