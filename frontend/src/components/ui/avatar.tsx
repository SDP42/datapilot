import { cn } from "@/lib/utils";

export function Avatar({ name, className, size = 36 }: { name: string; className?: string; size?: number }) {
  const initials = name
    .split(/[\s._-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0]?.toUpperCase())
    .join("");

  return (
    <div
      className={cn(
        "flex items-center justify-center rounded-full bg-gradient-to-br from-primary to-accent-2 font-semibold text-white shadow-md shadow-primary/30",
        className,
      )}
      style={{ width: size, height: size, fontSize: size * 0.4 }}
    >
      {initials || "?"}
    </div>
  );
}
