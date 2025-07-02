
import Header from '@/components/Header';
import Hero from '@/components/Hero';
import StorytellingSection from '@/components/StorytellingSection';
import Features from '@/components/Features';
import EmotionalQuote from '@/components/EmotionalQuote';
import Pricing from '@/components/Pricing';
import Footer from '@/components/Footer';

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <Hero />
      <StorytellingSection />
      <Features />
      <EmotionalQuote />
      <Pricing />
      <Footer />
    </div>
  );
};

export default Index;
