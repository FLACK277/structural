
import * as React from "react"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { LucideIcon } from "lucide-react"

interface FeatureCardProps extends React.HTMLAttributes<HTMLDivElement> {
  icon: LucideIcon;
  title: string;
  description: string;
}

const FeatureCard = React.forwardRef<HTMLDivElement, FeatureCardProps>(
  ({ className, icon: Icon, title, description, ...props }, ref) => {
    return (
      <Card
        ref={ref}
        className={cn(
          "glass dark:glass-dark hover:scale-105 transition-all duration-300 cursor-pointer group",
          className
        )}
        {...props}
      >
        <CardContent className="p-6 text-center space-y-4">
          <div className="w-12 h-12 mx-auto rounded-lg bg-gradient-primary flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
            <Icon className="h-6 w-6 text-white" />
          </div>
          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-foreground">{title}</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">{description}</p>
          </div>
        </CardContent>
      </Card>
    )
  }
)
FeatureCard.displayName = "FeatureCard"

export { FeatureCard }
