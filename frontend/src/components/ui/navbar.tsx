
import { useState } from 'react';
import { CTAButton } from '@/components/ui/cta-button';
import ThemeSwitch from '@/components/ui/theme-switch';
import { Menu, X } from 'lucide-react';

interface NavbarProps {
  logo?: string;
  links?: Array<{ href: string; label: string }>;
  showAuth?: boolean;
}

const Navbar = ({ 
  logo = "SkillBridge", 
  links = [
    { href: "#features", label: "Features" },
    { href: "#pricing", label: "Pricing" },
    { href: "#about", label: "About" }
  ],
  showAuth = true 
}: NavbarProps) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <header className="fixed top-0 w-full z-50 glass dark:glass-dark border-b border-border/20">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
            {logo}
          </div>
          
          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-8">
            {links.map((link) => (
              <a 
                key={link.href}
                href={link.href} 
                className="text-foreground hover:text-primary transition-colors duration-300 relative group"
              >
                {link.label}
                <span className="absolute inset-x-0 bottom-0 h-0.5 bg-primary scale-x-0 group-hover:scale-x-100 transition-transform duration-300" />
              </a>
            ))}
          </nav>
          
          {/* Desktop Actions */}
          <div className="hidden md:flex items-center space-x-4">
            <ThemeSwitch />
            {showAuth && (
              <>
                <CTAButton variant="secondary" size="sm">
                  Sign In
                </CTAButton>
                <CTAButton size="sm">
                  Get Started
                </CTAButton>
              </>
            )}
          </div>
          
          {/* Mobile Menu Button */}
          <div className="flex items-center space-x-2 md:hidden">
            <ThemeSwitch />
            <CTAButton
              variant="secondary"
              size="sm"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </CTAButton>
          </div>
        </div>
        
        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden mt-4 py-4 border-t border-border/20 glass dark:glass-dark rounded-lg animate-fade-in">
            <nav className="flex flex-col space-y-4">
              {links.map((link) => (
                <a 
                  key={link.href}
                  href={link.href} 
                  className="text-foreground hover:text-primary transition-colors duration-300 px-4 py-2"
                  onClick={() => setIsMenuOpen(false)}
                >
                  {link.label}
                </a>
              ))}
              {showAuth && (
                <div className="flex flex-col space-y-2 px-4 pt-4 border-t border-border/20">
                  <CTAButton variant="secondary" size="sm">
                    Sign In
                  </CTAButton>
                  <CTAButton size="sm">
                    Get Started
                  </CTAButton>
                </div>
              )}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default Navbar;
