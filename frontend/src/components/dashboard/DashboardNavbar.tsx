
import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Menu, Sun, Moon, Bell } from 'lucide-react';
import { useState, useEffect } from 'react';

interface DashboardNavbarProps {
  toggleSidebar: () => void;
  sidebarOpen: boolean;
}

const DashboardNavbar = ({ toggleSidebar, sidebarOpen }: DashboardNavbarProps) => {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    // Sync with global theme state
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const shouldUseDark = savedTheme === 'dark' || (!savedTheme && prefersDark);
    setIsDark(shouldUseDark);
  }, []);

  const toggleDarkMode = () => {
    const newMode = !isDark;
    setIsDark(newMode);
    localStorage.setItem('theme', newMode ? 'dark' : 'light');
    document.documentElement.classList.toggle('dark', newMode);
    
    // TODO: Backend Integration Point
    // Save theme preference to user profile
    // await updateUserPreferences({ theme: newMode ? 'dark' : 'light' });
  };

  return (
    <header className="h-16 border-b border-border bg-card/50 dark:bg-card/50 backdrop-blur-sm">
      <div className="flex items-center justify-between h-full px-6">
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSidebar}
            className="md:hidden"
          >
            <Menu className="h-4 w-4" />
          </Button>
          
          <div>
            <h1 className="text-xl font-semibold">Dashboard</h1>
          </div>
        </div>
        
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
          
          <Button variant="ghost" size="icon" className="rounded-full">
            <Bell className="h-4 w-4" />
          </Button>
          
          <Avatar className="h-8 w-8">
            <AvatarImage src="/placeholder.svg" />
            <AvatarFallback>RK</AvatarFallback>
          </Avatar>
        </div>
      </div>
    </header>
  );
};

export default DashboardNavbar;
