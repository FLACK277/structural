
import { useEffect, useState } from 'react';

const GlowingIllustration = () => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="relative w-full max-w-md mx-auto">
      {/* Animated Map of India with glowing effect */}
      <div className="relative">
        {/* Background glow */}
        <div className="absolute inset-0 animate-pulse">
          <div className="w-full h-64 bg-gradient-primary opacity-20 blur-3xl rounded-full"></div>
        </div>
        
        {/* Main illustration container */}
        <div className="relative z-10 flex items-center justify-center h-64">
          {/* Career Growth Steps */}
          <div className="relative w-48 h-48">
            {/* Glowing orb in center */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <div className="w-12 h-12 bg-gradient-primary rounded-full animate-float">
                <div className="w-full h-full bg-gradient-primary rounded-full animate-pulse opacity-75"></div>
              </div>
              <div className="absolute inset-0 w-12 h-12 bg-gradient-primary rounded-full blur-lg opacity-50 animate-pulse"></div>
            </div>
            
            {/* Connecting nodes representing career progression */}
            {[0, 1, 2, 3, 4].map((step) => {
              const angle = (step * 72 * Math.PI) / 180; // 72 degrees apart
              const radius = 60;
              const x = Math.cos(angle) * radius;
              const y = Math.sin(angle) * radius;
              
              return (
                <div
                  key={step}
                  className="absolute w-6 h-6 transform -translate-x-1/2 -translate-y-1/2"
                  style={{
                    left: `calc(50% + ${x}px)`,
                    top: `calc(50% + ${y}px)`,
                    animationDelay: `${step * 0.2}s`
                  }}
                >
                  <div className="w-full h-full bg-primary/60 rounded-full animate-ping"></div>
                  <div className="absolute inset-0 w-full h-full bg-primary rounded-full"></div>
                </div>
              );
            })}
            
            {/* Connecting lines */}
            {[0, 1, 2, 3, 4].map((step) => {
              const angle = (step * 72 * Math.PI) / 180;
              const nextAngle = ((step + 1) * 72 * Math.PI) / 180;
              const radius = 60;
              
              return (
                <div
                  key={`line-${step}`}
                  className="absolute w-px h-12 bg-gradient-to-b from-primary/50 to-transparent origin-bottom"
                  style={{
                    left: `calc(50% + ${Math.cos(angle) * radius / 2}px)`,
                    top: `calc(50% + ${Math.sin(angle) * radius / 2}px)`,
                    transform: `rotate(${angle * 180 / Math.PI + 90}deg)`,
                    animationDelay: `${step * 0.3}s`
                  }}
                />
              );
            })}
          </div>
        </div>
        
        {/* Floating particles */}
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 bg-primary/40 rounded-full animate-float"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${i * 0.5}s`,
              animationDuration: `${3 + Math.random() * 2}s`
            }}
          />
        ))}
      </div>
      
      {/* Skill icons floating around */}
      <div className="absolute inset-0 pointer-events-none">
        {['💻', '🎯', '🚀', '⭐', '🔗', '📈'].map((icon, index) => (
          <div
            key={index}
            className="absolute text-2xl animate-float opacity-60"
            style={{
              left: `${15 + (index * 15)}%`,
              top: `${10 + (index % 2) * 60}%`,
              animationDelay: `${index * 0.8}s`,
              animationDuration: '4s'
            }}
          >
            {icon}
          </div>
        ))}
      </div>
    </div>
  );
};

export default GlowingIllustration;
