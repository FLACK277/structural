import { useState, useEffect } from 'react';
import DashboardSidebar from '@/components/dashboard/DashboardSidebar';
import DashboardNavbar from '@/components/dashboard/DashboardNavbar';
import MyProfile from '@/components/dashboard/MyProfile';
import SkillAssessment from '@/components/dashboard/SkillAssessment';
import JobRecommendations from '@/components/dashboard/JobRecommendations';
import CareerInsights from '@/components/dashboard/CareerInsights';
import ResumeInsights from '@/components/dashboard/ResumeInsights';
import AssessmentOverview from '@/components/dashboard/AssessmentOverview';
import ScrollAnimation from '@/components/ui/scroll-animation';
import Header from '@/components/Header';
import { useAuth } from '@/lib/AuthContext';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('profile');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { token } = useAuth();
  const [profile, setProfile] = useState(null);
  const [profileLoading, setProfileLoading] = useState(true);
  const [profileError, setProfileError] = useState('');
  const [dashboardData, setDashboardData] = useState(null);
  const [dashboardLoading, setDashboardLoading] = useState(true);
  const [dashboardError, setDashboardError] = useState('');
  const [assessments, setAssessments] = useState([]);
  const [assessmentsLoading, setAssessmentsLoading] = useState(true);
  const [assessmentsError, setAssessmentsError] = useState('');

  useEffect(() => {
    const fetchProfile = async () => {
      setProfileLoading(true);
      setProfileError('');
      try {
        const res = await fetch('http://localhost:8000/api/me', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (!res.ok) throw new Error('Failed to fetch profile');
        const data = await res.json();
        setProfile(data);
      } catch (err) {
        setProfileError('Could not load profile');
      } finally {
        setProfileLoading(false);
      }
    };
    if (token) fetchProfile();
  }, [token]);

  useEffect(() => {
    const fetchDashboard = async (user_id) => {
      setDashboardLoading(true);
      setDashboardError('');
      try {
        const res = await fetch(`http://localhost:8000/api/dashboard/${user_id}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (!res.ok) throw new Error('Failed to fetch dashboard data');
        const data = await res.json();
        setDashboardData(data);
      } catch (err) {
        setDashboardError('Could not load dashboard data');
      } finally {
        setDashboardLoading(false);
      }
    };
    if (profile && profile.user_id) {
      fetchDashboard(profile.user_id);
    }
  }, [profile, token]);

  useEffect(() => {
    // Fetch assessments
    const fetchAssessments = async (user_id) => {
      setAssessmentsLoading(true);
      setAssessmentsError('');
      try {
        const res = await fetch(`http://localhost:8000/api/user_assessments/${user_id}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error('Failed to fetch assessments');
        const data = await res.json();
        setAssessments(data);
      } catch (err) {
        setAssessmentsError('Could not load assessments');
      } finally {
        setAssessmentsLoading(false);
      }
    };
    if (profile && profile.user_id) {
      fetchAssessments(profile.user_id);
    }
  }, [profile, token]);

  // Add a function to refresh both profile and dashboard data
  const refreshDashboard = async () => {
    if (!token) return;
    setProfileLoading(true);
    setDashboardLoading(true);
    setProfileError('');
    setDashboardError('');
    try {
      // Fetch profile
      const resProfile = await fetch('http://localhost:8000/api/me', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!resProfile.ok) throw new Error('Failed to fetch profile');
      const dataProfile = await resProfile.json();
      setProfile(dataProfile);
      // Fetch dashboard
      if (dataProfile && dataProfile.user_id) {
        const resDashboard = await fetch(`http://localhost:8000/api/dashboard/${dataProfile.user_id}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!resDashboard.ok) throw new Error('Failed to fetch dashboard data');
        const dataDashboard = await resDashboard.json();
        setDashboardData(dataDashboard);
      }
    } catch (err) {
      setProfileError('Could not refresh profile');
      setDashboardError('Could not refresh dashboard data');
    } finally {
      setProfileLoading(false);
      setDashboardLoading(false);
    }
  };

  // Assessment stats
  const totalAssessments = assessments.length;
  const averageScore = totalAssessments > 0
    ? Math.round((assessments.reduce((sum, a) => sum + (typeof a.score === 'number' ? a.score : 0), 0) / totalAssessments) * 100)
    : 0;
  // For completed, assume all fetched are completed; for total available, use a constant or count unique skills in question bank
  const completedAssessments = totalAssessments;
  const totalAvailableAssessments = 12; // You can make this dynamic if needed

  const renderContent = () => {
    switch (activeTab) {
      case 'profile':
        return (
          <div className="space-y-6">
            <ScrollAnimation animation="fade-in">
              <MyProfile dashboardData={dashboardData} refreshDashboard={refreshDashboard} userId={profile?.user_id} />
            </ScrollAnimation>
            <ScrollAnimation animation="slide-up" delay={200}>
              <ResumeInsights 
                education={dashboardData?.education}
                resume_skills={dashboardData?.resume_skills}
                career_goals={dashboardData?.career_goals}
                profile_completeness={dashboardData?.profile_completeness}
              />
            </ScrollAnimation>
          </div>
        );
      case 'assessment':
        return (
          <div className="space-y-6">
            <ScrollAnimation animation="fade-in">
              <SkillAssessment dashboardData={dashboardData} />
            </ScrollAnimation>
            <ScrollAnimation animation="slide-up" delay={200}>
              <AssessmentOverview />
            </ScrollAnimation>
          </div>
        );
      case 'jobs':
        return (
          <ScrollAnimation animation="fade-in">
            <JobRecommendations dashboardData={dashboardData} />
          </ScrollAnimation>
        );
      case 'insights':
        return (
          <ScrollAnimation animation="fade-in">
            <CareerInsights />
          </ScrollAnimation>
        );
      default:
        return (
          <ScrollAnimation animation="fade-in">
            <MyProfile />
          </ScrollAnimation>
        );
    }
  };

  return (
    <>
      <Header />
      <div className='mt-20'>
        <div className="min-h-screen bg-background dark:bg-background flex flex-col">
          {/* Welcome message and dashboard summary */}
          <div className="w-full px-8 pt-6 pb-2">
            {profileLoading ? (
              <div className="text-lg text-muted-foreground">Loading your profile...</div>
            ) : profileError ? (
              <div className="text-red-500">{profileError}</div>
            ) : profile ? (
              <div className="text-2xl font-semibold text-primary mb-2">
                Welcome, {profile.username} <span className="text-base text-muted-foreground">({profile.email})</span>
              </div>
            ) : null}
            {dashboardLoading ? (
              <div className="text-muted-foreground">Loading dashboard data...</div>
            ) : dashboardError ? (
              <div className="text-red-500">{dashboardError}</div>
            ) : dashboardData ? (
              <div className="mt-2 flex flex-wrap gap-8">
                {/* Top Skills */}
                {dashboardData.skills && dashboardData.skills.length > 0 && (
                  <div>
                    <div className="font-semibold text-lg mb-1">Top Skills</div>
                    <ul className="list-disc list-inside text-muted-foreground">
                      {dashboardData.skills.slice(0, 5).map((skill, i) => (
                        <li key={i}>{skill.name} (Level: {skill.level})</li>
                      ))}
                    </ul>
                  </div>
                )}
                {/* Learning Path */}
                {/* {dashboardData.learning_path && (
                  <div>
                    <div className="font-semibold text-lg mb-1">Learning Path Progress</div>
                    <div className="text-muted-foreground">
                      Skills Gap: {dashboardData.learning_path.skills_gap?.join(', ') || 'None'}<br />
                      Estimated Completion: {dashboardData.learning_path.estimated_completion} days
                    </div>
                  </div>
                )} */}
              </div>
            ) : null}
          </div>
          <div className="flex-1 flex">
            <DashboardSidebar 
              activeTab={activeTab} 
              setActiveTab={setActiveTab}
              isOpen={sidebarOpen}
              setIsOpen={setSidebarOpen}
              dashboardData={dashboardData}
            />
            <div className="flex-1 flex flex-col">
              <DashboardNavbar 
                toggleSidebar={() => setSidebarOpen(!sidebarOpen)}
                sidebarOpen={sidebarOpen}
              />
              <main className="flex-1 p-6 overflow-auto bg-background dark:bg-background">
                <div className="animate-fade-in">
                  {/* Assessment stats row */}
                  <div className="flex justify-center">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                      <div className="bg-card rounded-lg p-6 flex flex-col items-center">
                        <div className="text-2xl mb-2">📚</div>
                        <h3 className="font-semibold">Total Assessments</h3>
                        <p className="text-2xl font-bold">{totalAssessments}</p>
                      </div>
                      <div className="bg-card rounded-lg p-6 flex flex-col items-center">
                        <div className="text-2xl mb-2">🏆</div>
                        <h3 className="font-semibold">Average Score</h3>
                        <p className="text-2xl font-bold">{averageScore}%</p>
                      </div>
                      <div className="bg-card rounded-lg p-6 flex flex-col items-center">
                        <div className="text-2xl mb-2">🎯</div>
                        <h3 className="font-semibold">Completed</h3>
                        <p className="text-2xl font-bold">{completedAssessments}/{totalAvailableAssessments}</p>
                      </div>
                    </div>
                  </div>
                  {renderContent()}
                </div>
              </main>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Dashboard;
