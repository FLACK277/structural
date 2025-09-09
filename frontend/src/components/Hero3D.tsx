import { Button } from '@/components/ui/button';
import { ArrowRight, Sparkles } from 'lucide-react';
import { Link } from 'react-router-dom';
import GlowingIllustration from './GlowingIllustration';
import Scene3D from './3d/Scene3D';
import FloatingCube from './3d/FloatingCube';
import AnimatedSphere from './3d/AnimatedSphere';
import ParticleField from './3d/ParticleField';

const Hero3D = () => {
  return (
    <section className="min-h-screen flex items-center justify-center pt-20 px-6 relative overflow-hidden">
      {/* 3D Background */}
      <div className="absolute inset-0 z-0">
        <Scene3D className="w-full h-full" cameraPosition={[0, 0, 8]}>
          <ParticleField />
          <FloatingCube position={[-3, 2, -2]} color="#6366f1" size={0.8} />
          <FloatingCube position={[3, -1, -1]} color="#8b5cf6" size={0.6} />
          <AnimatedSphere position={[-2, -2, -3]} color="#ec4899" radius={0.4} />
          <AnimatedSphere position={[2, 2, -2]} color="#f59e0b" radius={0.5} />
          <AnimatedSphere position={[0, -3, -4]} color="#10b981" radius={0.3} />
        </Scene3D>
      </div>

      {/* Content */}
      <div className="container mx-auto relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Text Content */}
          <div className="animate-fade-in">
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-relaxed md:leading-[1.2]">
              <span className="bg-gradient-primary bg-clip-text">
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
            
            <div className="flex flex-col sm:flex-row gap-4">
              <Button 
                size="lg" 
                className="bg-gradient-primary text-white hover:opacity-90 text-lg px-8 py-6 rounded-full animate-float"
              >
                Start Your Journey
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
              
              <Link to="/demo">
                <Button 
                  size="lg" 
                  variant="outline"
                  className="text-lg px-8 py-6 rounded-full backdrop-blur-md border-white/20 hover:bg-white/10"
                >
                  <Sparkles className="mr-2 h-5 w-5" />
                  View 3D Demo
                </Button>
              </Link>
            </div>
          </div>
          
          {/* Glowing Illustration */}
          <div className="animate-slide-in flex justify-center lg:justify-end">
            <GlowingIllustration />
          </div>
        </div>
        
        {/* Stats Section */}
        <div className="mt-16 relative">
          <div className="glass dark:glass-dark rounded-3xl p-8 max-w-4xl mx-auto animate-slide-in backdrop-blur-md">
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

export default Hero3D;