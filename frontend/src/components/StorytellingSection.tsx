
const StorytellingSection = () => {
  return (
    <section className="py-20 px-6">
      <div className="container mx-auto max-w-4xl">
        <div className="space-y-16">
          <div className="animate-fade-in">
            <h2 className="text-4xl md:text-5xl font-bold text-center mb-12">
              The Indian Job Market Challenge
            </h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <div className="animate-slide-in">
              <div className="glass dark:glass-dark p-8 rounded-3xl">
                <div className="text-6xl font-bold text-destructive mb-4">65%</div>
                <div className="text-xl font-semibold mb-4">Skill-Job Mismatch</div>
                <p className="text-muted-foreground">
                  Two-thirds of Indian graduates struggle to find jobs that match their skills, 
                  leading to unemployment and underemployment.
                </p>
              </div>
            </div>
            
            <div className="animate-slide-in" style={{ animationDelay: '0.2s' }}>
              <div className="glass dark:glass-dark p-8 rounded-3xl">
                <div className="text-6xl font-bold text-orange-500 mb-4">70%</div>
                <div className="text-xl font-semibold mb-4">Engineering Pressure</div>
                <p className="text-muted-foreground">
                  Students are pushed into engineering streams without considering 
                  their natural talents and interests.
                </p>
              </div>
            </div>
          </div>
          
          <div className="text-center animate-fade-in" style={{ animationDelay: '0.4s' }}>
            <div className="glass dark:glass-dark p-12 rounded-3xl max-w-3xl mx-auto">
              <h3 className="text-3xl font-bold mb-6">The Rural Reality</h3>
              <p className="text-lg text-muted-foreground leading-relaxed">
                In rural India, talented individuals often lack access to career guidance 
                and opportunities. Traditional job search methods fail to recognize 
                their potential, creating a vast pool of untapped talent.
              </p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="animate-slide-in text-center" style={{ animationDelay: '0.6s' }}>
              <div className="glass dark:glass-dark p-6 rounded-2xl">
                <div className="text-4xl mb-4">🎓</div>
                <h4 className="font-semibold mb-2">Limited Guidance</h4>
                <p className="text-sm text-muted-foreground">
                  Lack of career counseling and skill assessment
                </p>
              </div>
            </div>
            
            <div className="animate-slide-in text-center" style={{ animationDelay: '0.8s' }}>
              <div className="glass dark:glass-dark p-6 rounded-2xl">
                <div className="text-4xl mb-4">🌍</div>
                <h4 className="font-semibold mb-2">Geographic Barriers</h4>
                <p className="text-sm text-muted-foreground">
                  Distance from job markets and opportunities
                </p>
              </div>
            </div>
            
            <div className="animate-slide-in text-center" style={{ animationDelay: '1s' }}>
              <div className="glass dark:glass-dark p-6 rounded-2xl">
                <div className="text-4xl mb-4">💼</div>
                <h4 className="font-semibold mb-2">Skill Recognition</h4>
                <p className="text-sm text-muted-foreground">
                  Hidden talents go unnoticed by traditional hiring
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default StorytellingSection;
