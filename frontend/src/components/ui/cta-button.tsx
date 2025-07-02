
import * as React from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { cva, type VariantProps } from "class-variance-authority"

const ctaButtonVariants = cva(
  "inline-flex items-center justify-center rounded-lg text-sm font-medium ring-offset-background transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:scale-105 hover:shadow-lg",
  {
    variants: {
      variant: {
        primary: "bg-gradient-primary text-white hover:opacity-90 shadow-lg",
        secondary: "glass dark:glass-dark text-foreground hover:bg-accent border border-border",
      },
      size: {
        default: "h-10 px-6 py-2",
        sm: "h-9 rounded-md px-4",
        lg: "h-12 rounded-lg px-8 text-base",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "default",
    },
  }
)

export interface CTAButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof ctaButtonVariants> {}

const CTAButton = React.forwardRef<HTMLButtonElement, CTAButtonProps>(
  ({ className, variant, size, children, ...props }, ref) => {
    return (
      <Button
        className={cn(ctaButtonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      >
        {children}
      </Button>
    )
  }
)
CTAButton.displayName = "CTAButton"

export { CTAButton, ctaButtonVariants }
