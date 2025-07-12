import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Play, Trophy, Target } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/lib/AuthContext';
import { api, endpoints } from '@/lib/api'; // Add this import

const SkillAssessment = ({ dashboardData }: { dashboardData?: any }) => {
  const navigate = useNavigate();
  const { user, token } = useAuth();
  const userId = user?.user_id;
  const [assessments, setAssessments] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchAssessments = async () => {
      if (!userId) return;
      setLoading(true);
      setError('');
      try {
        const data = await api.fetchWithAuth(`${endpoints.userAssessments}/${userId}`, token);
        setAssessments(data);
      } catch (err) {
        setError('Could not load assessments');
      } finally {
        setLoading(false);
      }
    };
    fetchAssessments();
  }, [userId, token]);

  // Compute stats
  const totalAssessments = assessments.length;
  const getScorePercent = (a) => {
    if (typeof a.estimated_level === 'number' && a.estimated_level <= 1) {
      return a.estimated_level * 100;
    }
    if (typeof a.score === 'number') {
      // If score is <= 1, treat as float; if > 1, treat as percent
      return a.score <= 1 ? a.score * 100 : a.score;
    }
    return 0;
  };
  
  console.log('Assessments for average:', assessments);
  
  const averageScore = totalAssessments > 0
    ? Math.round((assessments.reduce((sum, a) => sum + (typeof a.score === 'number' ? a.score : 0), 0) / totalAssessments) * 100)
    : 0;
  console.log('averageScore (final, for display):', averageScore);
  const recentAssessments = assessments.slice(-5).reverse();

  console.log('Average calculation:', {
    totalAssessments,
    sum: assessments.reduce((sum, a) => sum + (typeof a.estimated_level === 'number' ? a.estimated_level : 0), 0),
    average: totalAssessments > 0
      ? assessments.reduce((sum, a) => sum + (typeof a.estimated_level === 'number' ? a.estimated_level : 0), 0) / totalAssessments
      : 0
  });

  // Get user's skills (top_skills or skills)
  const userSkills = dashboardData?.top_skills || dashboardData?.skills || [];
  // Get job requirements from learning_path.skills_gap
  const jobRequirements = (dashboardData?.learning_path?.skills_gap || []).map((name: string) => ({ name, required_level: 0.8 }));

  // Debug logging
  console.log('SkillAssessment - dashboardData:', dashboardData);
  console.log('SkillAssessment - learning_path:', dashboardData?.learning_path);
  console.log('SkillAssessment - skills_gap:', dashboardData?.learning_path?.skills_gap);
  console.log('SkillAssessment - jobRequirements:', jobRequirements);
  console.log('SkillAssessment - userSkills:', userSkills);

  // Compute skill gaps
  const skillGaps = (dashboardData?.learning_path?.skills_gap || []).map((gap: any) => {
    // Use backend-provided current_level and required_level, fallback to userSkills if needed
    const userSkill = userSkills.find((s: any) => s.name?.toLowerCase() === gap.name?.toLowerCase());
    const current = gap.current_level !== undefined
      ? Math.round(gap.current_level * 100)
      : userSkill
        ? Math.round(userSkill.level * 100)
        : 0;
    const required = gap.required_level !== undefined
      ? Math.round(gap.required_level * 100)
      : 90;
    return {
      skill: gap.name,
      current,
      required,
      gap: Math.max(0, required - current),
    };
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold">Skill Assessment</h2>
        <Button className="bg-gradient-primary" onClick={() => navigate('/assessment')}>
          <Play className="h-4 w-4 mr-2" />
          Start New Assessment
        </Button>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Assessments */}
        <Card className="glass dark:glass-dark">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Trophy className="h-5 w-5 mr-2" />
              Recent Assessments
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {loading ? (
                <div>Loading assessments...</div>
              ) : error ? (
                <div className="text-red-500">{error}</div>
              ) : recentAssessments.length > 0 ? recentAssessments.map((assessment, index) => (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                  <div>
                    <h4 className="font-medium">{assessment.skill}</h4>
                    <p className="text-sm text-muted-foreground">{assessment.created_at ? new Date(assessment.created_at).toLocaleString() : ''}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-semibold text-primary">{Math.round((assessment.estimated_level || assessment.score || 0) * 100)}%</div>
                  </div>
                </div>
              )) : <div className="text-muted-foreground">No assessments yet.</div>}
            </div>
          </CardContent>
        </Card>

        {/* Skill Gap Analysis */}
        <Card className="glass dark:glass-dark">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Target className="h-5 w-5 mr-2" />
              Skill Gap vs Job Requirements
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {skillGaps.length > 0 ? skillGaps.map((item, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{item.skill}</span>
                    <span className="text-muted-foreground">
                      {item.current}% / {item.required}%
                    </span>
                  </div>
                  <div className="relative">
                    <Progress value={item.current} className="h-2" />
                    <div 
                      className="absolute top-0 h-2 bg-red-500/20 rounded-full"
                      style={{ 
                        left: `${item.current}%`, 
                        width: `${item.gap}%` 
                      }}
                    />
                  </div>
                  {item.gap > 15 && (
                    <p className="text-xs text-red-500">
                      Gap: {item.gap}% - Needs improvement
                    </p>
                  )}
                </div>
              )) : <div className="text-muted-foreground">No skill gaps found.</div>}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Assessment Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="glass dark:glass-dark cursor-pointer hover:scale-105 transition-transform">
          <CardContent className="p-4 text-center">
            <div className="text-2xl mb-2">📚</div>
            <h3 className="font-semibold">Total Assessments</h3>
            <p className="text-2xl font-bold">{totalAssessments}</p>
          </CardContent>
        </Card>
        <Card className="glass dark:glass-dark cursor-pointer hover:scale-105 transition-transform">
          <CardContent className="p-4 text-center">
            <div className="text-2xl mb-2">🏆</div>
            <h3 className="font-semibold">Average Score</h3>
            <div>DEBUG: averageScore = {averageScore}</div>
            <p className="text-2xl font-bold">{averageScore}%</p>
          </CardContent>
        </Card>
        {/* You can add more stats here if needed */}
      </div>
    </div>
  );
};

export default SkillAssessment;