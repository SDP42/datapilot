import { HTMLAttributes } from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";
import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";

const alertVariants = cva("flex items-start gap-3 rounded-xl border p-4 text-sm", {
  variants: {
    variant: {
      info: "border-primary/30 bg-primary/10 text-foreground",
      success: "border-success/30 bg-success/10 text-foreground",
      warning: "border-warning/30 bg-warning/10 text-foreground",
      danger: "border-danger/30 bg-danger/10 text-foreground",
    },
  },
  defaultVariants: { variant: "info" },
});

const ICONS = { info: Info, success: CheckCircle2, warning: AlertTriangle, danger: XCircle };

export interface AlertProps extends HTMLAttributes<HTMLDivElement>, VariantProps<typeof alertVariants> {
  title?: string;
}

export function Alert({ className, variant = "info", title, children, ...props }: AlertProps) {
  const Icon = ICONS[variant ?? "info"];
  return (
    <div className={cn(alertVariants({ variant }), className)} {...props}>
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <div className="space-y-0.5">
        {title && <p className="font-medium">{title}</p>}
        <div className="text-muted">{children}</div>
      </div>
    </div>
  );
}
