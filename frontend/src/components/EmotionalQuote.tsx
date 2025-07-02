
import { Button } from '@/components/ui/button';
import { ArrowRight } from 'lucide-react';

const EmotionalQuote = () => {
  return (
    <section className="py-20 px-6 bg-gradient-to-br from-primary/10 to-purple-500/10">
      <div className="container mx-auto text-center">
        <div className="max-w-4xl mx-auto">
          <div className="animate-fade-in">
            <h2 className="text-5xl md:text-7xl font-bold mb-8 leading-tight">
              <span className="bg-gradient-primary bg-clip-text text-transparent">
                "You're not unskilled."
              </span>
            </h2>
            
            <p className="text-xl md:text-2xl text-muted-foreground mb-12 leading-relaxed">
              You have unique talents, experiences, and potential. Traditional hiring just 
              hasn't recognized them yet. It's time to change that narrative.
            </p>
            
            <div className="glass dark:glass-dark p-8 rounded-3xl mb-12 animate-slide-in">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div>
                  <div className="text-3xl font-bold text-primary mb-2">Every skill</div>
                  <div className="text-muted-foreground">has value</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-primary mb-2">Every person</div>
                  <div className="text-muted-foreground">has potential</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-primary mb-2">Every dream</div>
                  <div className="text-muted-foreground">deserves a chance</div>
                </div>
              </div>
            </div>
            
            <Button 
              size="lg" 
              className="bg-gradient-primary text-white hover:opacity-90 text-lg px-8 py-6 rounded-full animate-float"
            >
              Let SkillBridge Prove It
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default EmotionalQuote;
