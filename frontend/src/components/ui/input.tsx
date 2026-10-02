import { InputHTMLAttributes, forwardRef } from "react";
import { cn } from "@/lib/utils";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  error?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, error, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={cn(
          "flex h-12 w-full rounded-xl border bg-surface-2/60 px-4 text-base text-foreground placeholder:text-muted transition-colors outline-none",
          "border-surface-border focus:border-primary focus:ring-2 focus:ring-primary/25",
          error && "border-danger focus:border-danger focus:ring-danger/25",
          props.disabled && "opacity-50 cursor-not-allowed",
          className,
        )}
        {...props}
      />
    );
  },
);
Input.displayName = "Input";
