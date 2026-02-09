import Header from '@/components/Header';
import Hero3D from '@/components/Hero3D';
import StorytellingSection from '@/components/StorytellingSection';
import Features from '@/components/Features';
import EmotionalQuote from '@/components/EmotionalQuote';
import Pricing from '@/components/Pricing';
import Footer from '@/components/Footer';
import SmoothScroll from '@/components/SmoothScroll';

const Index3D = () => {
  return (
    <div className="min-h-screen bg-background">
      <SmoothScroll>
        <Header />
        <div data-scroll-section>
          <Hero3D />
        </div>
        <div data-scroll-section>
          <StorytellingSection />
        </div>
        <div data-scroll-section>
          <Features />
        </div>
        <div data-scroll-section>
          <EmotionalQuote />
        </div>
        <div data-scroll-section>
          <Pricing />
        </div>
        <div data-scroll-section>
          <Footer />
        </div>
      </SmoothScroll>
    </div>
  );
};

export default Index3D;