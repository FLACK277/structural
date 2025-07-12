import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Sun, Moon, Menu, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/lib/AuthContext';

const Header = () => {
  const [isDark, setIsDark] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const navigate = useNavigate();
  const { isAuthenticated, user, logout } = useAuth();

  useEffect(() => {
    // TODO: Backend Integration Point
    // Get theme preference from user profile if authenticated
    // For now, using localStorage but should sync with user preferences API
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    const shouldUseDark = savedTheme === 'dark' || (!savedTheme && prefersDark);
    setIsDark(shouldUseDark);
    document.documentElement.classList.toggle('dark', shouldUseDark);
  }, []);

  const toggleDarkMode = () => {
    const newMode = !isDark;
    setIsDark(newMode);
    localStorage.setItem('theme', newMode ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark', newMode);
    
    // TODO: Backend Integration Point
    // Save theme preference to user profile if authenticated
    // await updateUserPreferences({ theme: newMode ? 'dark' : 'light' });
  };

  const handleSignIn = () => {
    // TODO: Backend Integration Point
    // Replace with actual authentication flow
    // For demo purposes, simulate sign-in and navigate to dashboard
    navigate('/login');
  };

  const handleGetStarted = () => {
    // TODO: Backend Integration Point
    // This should trigger registration flow or onboarding
    // Could navigate to role selection (student, professional, recruiter)
    navigate('/register');
  };

  return (
    <header className="fixed top-0 w-full z-50 glass dark:glass-dark border-b border-border/20">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="text-2xl font-bold bg-gradient-primary bg-clip-text">
            SkillBridge
          </div>
          
          <nav className="hidden md:flex space-x-8">
            <a href="#features" className="text-foreground hover:text-primary transition-colors">Features</a>
            <a href="#pricing" className="text-foreground hover:text-primary transition-colors">Pricing</a>
            <a href="#about" className="text-foreground hover:text-primary transition-colors">About</a>
          </nav>
          
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleDarkMode}
              className="rounded-full transition-all duration-300 hover:scale-110"
            >
              {isDark ? (
                <Sun className="h-4 w-4 transition-transform duration-300" />
              ) : (
                <Moon className="h-4 w-4 transition-transform duration-300" />
              )}
            </Button>
            
            {isAuthenticated ? (
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-foreground">{user?.username}</span>
                <Button
                  variant="outline"
                  className="glass dark:glass-dark transition-all duration-300 hover:scale-105"
                  onClick={logout}
                >
                  Logout
                </Button>
              </div>
            ) : (
              <div className="hidden md:flex space-x-2">
                <Button 
                  variant="outline" 
                  className="glass dark:glass-dark transition-all duration-300 hover:scale-105"
                  onClick={handleSignIn}
                >
                  Sign In
                </Button>
                <Button 
                  className="bg-gradient-primary text-white hover:opacity-90 transition-all duration-300 hover:scale-105"
                  onClick={handleGetStarted}
                >
                  Get Started
                </Button>
              </div>
            )}
            
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              {isMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </Button>
          </div>
        </div>
        
        {isMenuOpen && (
          <div className="md:hidden mt-4 py-4 border-t glass dark:glass-dark rounded-lg animate-fade-in">
            <nav className="flex flex-col space-y-4">
              <a href="#features" className="text-foreground hover:text-primary transition-colors">Features</a>
              <a href="#pricing" className="text-foreground hover:text-primary transition-colors">Pricing</a>
              <a href="#about" className="text-foreground hover:text-primary transition-colors">About</a>
              <div className="flex flex-col space-y-2 pt-4">
                {isAuthenticated ? (
                  <Button 
                    variant="outline" 
                    className="glass dark:glass-dark"
                    onClick={logout}
                  >
                    Logout
                  </Button>
                ) : (
                  <>
                    <Button 
                      variant="outline" 
                      className="glass dark:glass-dark"
                      onClick={handleSignIn}
                    >
                      Sign In
                    </Button>
                    <Button 
                      className="bg-gradient-primary text-white hover:opacity-90"
                      onClick={handleGetStarted}
                    >
                      Get Started
                    </Button>
                  </>
                )}
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
