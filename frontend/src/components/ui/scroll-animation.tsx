
import React, { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

interface ScrollAnimationProps {
  children: React.ReactNode;
  animation?: 'fade-in' | 'slide-up' | 'slide-left' | 'slide-right' | 'scale-in';
  delay?: number;
  duration?: number;
  className?: string;
}

const ScrollAnimation = ({ 
  children, 
  animation = 'fade-in', 
  delay = 0, 
  duration = 600,
  className 
}: ScrollAnimationProps) => {
  const [isVisible, setIsVisible] = useState(false);
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setTimeout(() => setIsVisible(true), delay);
        }
      },
      { threshold: 0.1 }
    );

    const currentElement = elementRef.current;
    if (currentElement) {
      observer.observe(currentElement);
    }

    return () => {
      if (currentElement) {
        observer.unobserve(currentElement);
      }
    };
  }, [delay]);

  const animationClasses = {
    'fade-in': isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8',
    'slide-up': isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-12',
    'slide-left': isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8',
    'slide-right': isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-8',
    'scale-in': isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
  };

  return (
    <div
      ref={elementRef}
      className={cn(
        'transition-all ease-out',
        animationClasses[animation],
        className
      )}
      style={{ transitionDuration: `${duration}ms` }}
    >
      {children}
    </div>
  );
};

export default ScrollAnimation;
