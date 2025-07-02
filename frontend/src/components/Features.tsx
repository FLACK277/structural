
import { Brain, Target, BookOpen, Users, Globe, Smartphone } from 'lucide-react';

const Features = () => {
  const features = [
    {
      icon: Brain,
      title: "AI Skill Assessments",
      description: "Advanced algorithms analyze your true capabilities beyond traditional qualifications",
      color: "text-blue-500"
    },
    {
      icon: Target,
      title: "Smart Job Matching",
      description: "Precision matching based on skills, personality, and career aspirations",
      color: "text-green-500"
    },
    {
      icon: BookOpen,
      title: "Personalized Learning",
      description: "Custom learning paths to bridge skill gaps and accelerate career growth",
      color: "text-purple-500"
    },
    {
      icon: Users,
      title: "Rural Support",
      description: "Dedicated support for rural talent with local language assistance",
      color: "text-orange-500"
    },
    {
      icon: Globe,
      title: "Multi-Language",
      description: "Platform available in 12+ Indian languages for wider accessibility",
      color: "text-pink-500"
    },
    {
      icon: Smartphone,
      title: "Mobile-First",
      description: "Optimized for smartphones to reach users across India",
      color: "text-indigo-500"
    }
  ];

  return (
    <section id="features" className="py-20 px-6">
      <div className="container mx-auto">
        <div className="text-center mb-16 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Powered by AI, Built for India
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Our comprehensive platform addresses the unique challenges of the Indian job market 
            with cutting-edge technology and local insights.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={feature.title}
              className="animate-fade-in glass dark:glass-dark p-8 rounded-3xl hover:scale-105 transition-all duration-300"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className={`${feature.color} mb-6`}>
                <feature.icon className="h-12 w-12" />
              </div>
              <h3 className="text-xl font-semibold mb-4">{feature.title}</h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
