import { Button } from '@/components/ui/button';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { User, BarChart, Briefcase, TrendingUp, Menu, X } from 'lucide-react';

type DashboardSidebarProps = {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
  dashboardData?: any;
};

const DashboardSidebar = ({ activeTab, setActiveTab, isOpen, setIsOpen, dashboardData }: DashboardSidebarProps) => {
  const menuItems = [
    { id: 'profile', label: 'My Profile', icon: User },
    { id: 'assessment', label: 'Skill Assessment', icon: BarChart },
    { id: 'jobs', label: 'Job Recommendations', icon: Briefcase },
    { id: 'insights', label: 'Career Insights', icon: TrendingUp },
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed md:relative top-0 left-0 h-full w-64 bg-card border-r border-border z-50
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        ${!isOpen && 'md:w-16'}
      `}>
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between">
            <div className={`flex items-center space-x-3 ${!isOpen && 'md:justify-center'}`}>
              <Avatar className="h-8 w-8">
                <AvatarImage src="/placeholder.svg" />
                <AvatarFallback>RK</AvatarFallback>
              </Avatar>
              {(isOpen || window.innerWidth < 768) && (
                <div>
                  <p className="font-medium">{dashboardData?.name || 'Your Name'}</p>
                  <p className="text-sm text-muted-foreground">{dashboardData?.headline || 'Your Headline'}</p>
                </div>
              )}
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setIsOpen(false)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        <nav className="p-4 space-y-2">
          {menuItems.map((item) => (
            <Button
              key={item.id}
              variant={activeTab === item.id ? "default" : "ghost"}
              className={`w-full justify-start ${!isOpen && 'md:justify-center md:px-2'}`}
              onClick={() => setActiveTab(item.id)}
            >
              <item.icon className="h-4 w-4" />
              {(isOpen || window.innerWidth < 768) && (
                <span className="ml-2">{item.label}</span>
              )}
            </Button>
          ))}
        </nav>
      </div>
    </>
  );
};

export default DashboardSidebar;
