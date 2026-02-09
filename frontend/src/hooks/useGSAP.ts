import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export const useGSAPAnimation = (animation: (element: HTMLElement) => void, dependencies: any[] = []) => {
  const elementRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (elementRef.current) {
      animation(elementRef.current);
    }
  }, dependencies);

  return elementRef;
};

export const useFadeInUp = (delay: number = 0) => {
  return useGSAPAnimation((element) => {
    gsap.fromTo(element, 
      { 
        opacity: 0, 
        y: 50 
      },
      { 
        opacity: 1, 
        y: 0, 
        duration: 1, 
        delay,
        ease: "power2.out",
        scrollTrigger: {
          trigger: element,
          start: "top 80%",
          toggleActions: "play none none reverse"
        }
      }
    );
  });
};

export const useFadeInLeft = (delay: number = 0) => {
  return useGSAPAnimation((element) => {
    gsap.fromTo(element, 
      { 
        opacity: 0, 
        x: -50 
      },
      { 
        opacity: 1, 
        x: 0, 
        duration: 1, 
        delay,
        ease: "power2.out",
        scrollTrigger: {
          trigger: element,
          start: "top 80%",
          toggleActions: "play none none reverse"
        }
      }
    );
  });
};

export const useFadeInRight = (delay: number = 0) => {
  return useGSAPAnimation((element) => {
    gsap.fromTo(element, 
      { 
        opacity: 0, 
        x: 50 
      },
      { 
        opacity: 1, 
        x: 0, 
        duration: 1, 
        delay,
        ease: "power2.out",
        scrollTrigger: {
          trigger: element,
          start: "top 80%",
          toggleActions: "play none none reverse"
        }
      }
    );
  });
};

export const useScaleIn = (delay: number = 0) => {
  return useGSAPAnimation((element) => {
    gsap.fromTo(element, 
      { 
        opacity: 0, 
        scale: 0.8 
      },
      { 
        opacity: 1, 
        scale: 1, 
        duration: 1, 
        delay,
        ease: "back.out(1.7)",
        scrollTrigger: {
          trigger: element,
          start: "top 80%",
          toggleActions: "play none none reverse"
        }
      }
    );
  });
};