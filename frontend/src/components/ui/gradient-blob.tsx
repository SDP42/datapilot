import { cn } from "@/lib/utils";

export function GradientBlob({ className }: { className?: string }) {
  return (
    <div className={cn("pointer-events-none absolute", className)}>
      <div className="h-56 w-56 animate-float-slow rounded-full bg-gradient-to-br from-primary/30 via-accent-2/20 to-accent/20 blur-[60px] sm:h-80 sm:w-80 sm:blur-[80px] lg:h-[32rem] lg:w-[32rem] lg:blur-[100px]" />
    </div>
  );
}
