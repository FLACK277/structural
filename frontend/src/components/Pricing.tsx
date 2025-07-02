
import { Button } from '@/components/ui/button';
import { Check } from 'lucide-react';

const Pricing = () => {
  const plans = [
    {
      name: "Free",
      price: "₹0",
      period: "/month",
      description: "Perfect for getting started",
      features: [
        "Basic skill assessments",
        "Job matching algorithm",
        "Profile creation",
        "Limited job applications",
        "Community access"
      ],
      cta: "Get Started",
      popular: false
    },
    {
      name: "Premium",
      price: "₹799",
      period: "/month",
      description: "For serious job seekers",
      features: [
        "Advanced AI assessments",
        "AI career chatbot",
        "Resume parsing & optimization",
        "Detailed insights & analytics",
        "Priority job matching",
        "Personalized learning paths",
        "Direct recruiter contact",
        "Interview preparation"
      ],
      cta: "Start Premium",
      popular: true
    },
    {
      name: "Enterprise",
      price: "Custom",
      period: "pricing",
      description: "For recruiters and organizations",
      features: [
        "Bulk candidate screening",
        "Custom assessment creation",
        "Advanced analytics dashboard",
        "API access",
        "Dedicated account manager",
        "White-label solutions",
        "Integration support",
        "Priority technical support"
      ],
      cta: "Contact Sales",
      popular: false
    }
  ];

  return (
    <section id="pricing" className="py-20 px-6">
      <div className="container mx-auto">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Choose Your Path to Success
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Flexible pricing options designed for every stage of your career journey, 
            from students to enterprise recruiters.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {plans.map((plan, index) => (
            <div 
              key={plan.name}
              className={`animate-fade-in glass dark:glass-dark p-8 rounded-3xl relative ${
                plan.popular ? 'ring-2 ring-primary scale-105' : ''
              }`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {plan.popular && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <span className="bg-gradient-primary text-white px-4 py-1 rounded-full text-sm font-medium">
                    Most Popular
                  </span>
                </div>
              )}
              
              <div className="text-center mb-8">
                <h3 className="text-2xl font-bold mb-2">{plan.name}</h3>
                <div className="text-4xl font-bold mb-2">
                  {plan.price}
                  <span className="text-lg text-muted-foreground">{plan.period}</span>
                </div>
                <p className="text-muted-foreground">{plan.description}</p>
              </div>
              
              <ul className="space-y-4 mb-8">
                {plan.features.map((feature, featureIndex) => (
                  <li key={featureIndex} className="flex items-start space-x-3">
                    <Check className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                    <span className="text-sm">{feature}</span>
                  </li>
                ))}
              </ul>
              
              <Button 
                className={`w-full ${
                  plan.popular 
                    ? 'bg-gradient-primary text-white hover:opacity-90' 
                    : 'glass dark:glass-dark hover:bg-primary hover:text-white'
                }`}
                size="lg"
              >
                {plan.cta}
              </Button>
            </div>
          ))}
        </div>
        
        <div className="text-center mt-12 animate-fade-in" style={{ animationDelay: '0.4s' }}>
          <p className="text-muted-foreground">
            All plans include 24/7 support and a 30-day money-back guarantee
          </p>
        </div>
      </div>
    </section>
  );
};

export default Pricing;
