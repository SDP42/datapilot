import { cn } from "@/lib/utils";

export function ShinyText({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "inline-block bg-clip-text text-transparent animate-shimmer",
        className,
      )}
      style={{
        backgroundImage:
          "linear-gradient(110deg, #8b90a8 35%, #ffffff 50%, #8b90a8 65%)",
        backgroundSize: "200% 100%",
      }}
    >
      {children}
    </span>
  );
}
