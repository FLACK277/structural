
import * as React from "react"
import { cn } from "@/lib/utils"

interface SectionWrapperProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "glass" | "gradient";
  children: React.ReactNode;
}

const SectionWrapper = React.forwardRef<HTMLDivElement, SectionWrapperProps>(
  ({ className, variant = "default", children, ...props }, ref) => {
    const variants = {
      default: "bg-background",
      glass: "glass dark:glass-dark",
      gradient: "bg-gradient-secondary dark:bg-gradient-primary",
    };

    return (
      <div
        ref={ref}
        className={cn(
          "py-16 px-6",
          variants[variant],
          className
        )}
        {...props}
      >
        <div className="container mx-auto max-w-6xl">
          {children}
        </div>
      </div>
    )
  }
)
SectionWrapper.displayName = "SectionWrapper"

export { SectionWrapper }
