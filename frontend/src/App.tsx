import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useState, useEffect } from "react";
import Index from "./pages/Index";
import Dashboard from "./pages/Dashboard";
import RecruiterDashboard from "./pages/RecruiterDashboard";
import NotFound from "./pages/NotFound";
import Login from "./pages/Login";
import Register from "./pages/Register";
import { AuthProvider, useAuth } from "@/lib/AuthContext";
import ProtectedRoute from "@/components/ProtectedRoute";
import Assessment from './pages/Assessment';

const queryClient = new QueryClient();

// Simple loading component
const PageLoader = () => (
  <div className="min-h-screen bg-background dark:bg-background flex items-center justify-center">
    <div className="glass dark:glass-dark p-8 rounded-lg">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
      <p className="text-sm text-muted-foreground">Loading...</p>
    </div>
  </div>
);

// HomeRedirect component for root path
const HomeRedirect = () => {
  const { isAuthenticated } = useAuth();
  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }
  return <Index />;
};

const App = () => {
  const [isLoading, setIsLoading] = useState(false);

  // TODO: Backend Integration Point
  // This is where you'd implement:
  // 1. Authentication state management
  // 2. User role detection (student, employee, recruiter)
  // 3. Theme preference loading from user profile
  // 4. Route protection based on auth state
  
  useEffect(() => {
    // Simulate route loading - replace with actual auth checks
    const handleRouteChange = () => {
      setIsLoading(true);
      setTimeout(() => setIsLoading(false), 500);
    };

    // Listen for route changes if needed
    // This is just for demo - you might want to use React Router's built-in loading states
  }, []);

  if (isLoading) {
    return <PageLoader />;
  }

  return (
    <BrowserRouter>
      <AuthProvider>
        <QueryClientProvider client={queryClient}>
          <TooltipProvider>
            <Toaster />
            <Sonner />
            <Routes>
              <Route path="/" element={<HomeRedirect />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route path="/dashboard" element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              } />
              <Route path="/recruiter" element={
                <ProtectedRoute>
                  <RecruiterDashboard />
                </ProtectedRoute>
              } />
              <Route path="/assessment" element={<Assessment />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </TooltipProvider>
        </QueryClientProvider>
      </AuthProvider>
    </BrowserRouter>
  );
};

export default App;
