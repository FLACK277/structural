
import { Button } from '@/components/ui/button';
import { ArrowRight } from 'lucide-react';
import GlowingIllustration from './GlowingIllustration';

const Hero = () => {
  return (
    <section className="min-h-screen flex items-center justify-center pt-20 px-6">
      <div className="container mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Text Content */}
          <div className="animate-fade-in">
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="bg-gradient-primary bg-clip-text text-transparent">
                Bridging the gap
              </span>
              <br />
              between talent and opportunity
              <br />
              <span className="text-muted-foreground text-3xl md:text-4xl">
                in India.
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-2xl">
              AI-powered career platform connecting Indian students and professionals 
              with the right opportunities through intelligent skill matching.
            </p>
            
            <Button 
              size="lg" 
              className="bg-gradient-primary text-white hover:opacity-90 text-lg px-8 py-6 rounded-full animate-float"
            >
              Start Your Journey
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
          
          {/* Glowing Illustration */}
          <div className="animate-slide-in flex justify-center lg:justify-end">
            <GlowingIllustration />
          </div>
        </div>
        
        {/* Stats Section */}
        <div className="mt-16 relative">
          <div className="glass dark:glass-dark rounded-3xl p-8 max-w-4xl mx-auto animate-slide-in">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-3xl font-bold text-primary mb-2">1M+</div>
                <div className="text-muted-foreground">Students Registered</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary mb-2">50K+</div>
                <div className="text-muted-foreground">Jobs Matched</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-primary mb-2">95%</div>
                <div className="text-muted-foreground">Success Rate</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
