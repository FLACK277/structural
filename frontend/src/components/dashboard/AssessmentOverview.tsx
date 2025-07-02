
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { BookOpen, Trophy, Clock, Target } from 'lucide-react';

const AssessmentOverview = () => {
  // TODO: Backend Integration Point
  // This data should come from your assessment/quiz API
  // Replace with actual API call: const assessmentData = await fetchAssessmentResults(userId);
  const assessmentData = {
    totalAssessments: 12,
    completedAssessments: 8,
    averageScore: 78,
    timeSpent: 145, // minutes
    skillBreakdown: [
      { skill: 'React', score: 85, color: '#3b82f6' },
      { skill: 'Python', score: 78, color: '#10b981' },
      { skill: 'SQL', score: 65, color: '#f59e0b' },
      { skill: 'ML', score: 72, color: '#8b5cf6' }
    ],
    recentAssessments: [
      { name: 'JavaScript Fundamentals', score: 85, date: '2024-01-15', status: 'completed' },
      { name: 'React Advanced', score: 78, date: '2024-01-10', status: 'completed' },
      { name: 'System Design', score: 0, date: '2024-01-20', status: 'pending' }
    ]
  };

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];

  return (
    <div className="space-y-6">
      {/* Assessment Stats */}
      {/* <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="glass dark:glass-dark border-border/20">
          <CardContent className="p-4 text-center">
            <BookOpen className="h-8 w-8 text-primary mx-auto mb-2" />
            <p className="text-2xl font-bold">{assessmentData.totalAssessments}</p>
            <p className="text-sm text-muted-foreground">Total Assessments</p>
          </CardContent>
        </Card>
        
        <Card className="glass dark:glass-dark border-border/20">
          <CardContent className="p-4 text-center">
            <Trophy className="h-8 w-8 text-green-500 mx-auto mb-2" />
            <p className="text-2xl font-bold">{assessmentData.averageScore}%</p>
            <p className="text-sm text-muted-foreground">Average Score</p>
          </CardContent>
        </Card>
        
        <Card className="glass dark:glass-dark border-border/20">
          <CardContent className="p-4 text-center">
            <Clock className="h-8 w-8 text-blue-500 mx-auto mb-2" />
            <p className="text-2xl font-bold">{assessmentData.timeSpent}m</p>
            <p className="text-sm text-muted-foreground">Time Invested</p>
          </CardContent>
        </Card>
        
        <Card className="glass dark:glass-dark border-border/20">
          <CardContent className="p-4 text-center">
            <Target className="h-8 w-8 text-purple-500 mx-auto mb-2" />
            <p className="text-2xl font-bold">{assessmentData.completedAssessments}/{assessmentData.totalAssessments}</p>
            <p className="text-sm text-muted-foreground">Completed</p>
          </CardContent>
        </Card>
      </div> */}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Skill Breakdown Chart */}
        {/* <Card className="glass dark:glass-dark border-border/20">
          <CardHeader>
            <CardTitle>Skill Assessment Scores</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={assessmentData.skillBreakdown}>
                <XAxis dataKey="skill" />
                <YAxis />
                <Bar dataKey="score" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card> */}

        {/* Recent Assessments */}
        {/* <Card className="glass dark:glass-dark border-border/20">
          <CardHeader>
            <CardTitle>Recent Assessments</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {assessmentData.recentAssessments.map((assessment, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-muted/50 dark:bg-muted/20 rounded-lg">
                <div className="flex-1">
                  <p className="font-medium text-sm">{assessment.name}</p>
                  <p className="text-xs text-muted-foreground">{assessment.date}</p>
                </div>
                <div className="flex items-center space-x-2">
                  {assessment.status === 'completed' ? (
                    <Badge variant="secondary">{assessment.score}%</Badge>
                  ) : (
                    <Button size="sm" variant="outline">Start</Button>
                  )}
                </div>
              </div>
            ))}
          </CardContent>
        </Card> */}
      </div>

      {/* TODO: Backend Integration Points */}
      {/* 
        API Integration needed for:
        1. fetchAssessmentResults(userId) - Get user's assessment history
        2. fetchSkillBreakdown(userId) - Get detailed skill analysis
        3. startAssessment(assessmentId) - Initialize new assessment
        4. getRecommendedAssessments(userId, userRole) - Get personalized suggestions
        5. Real-time score updates during assessment
      */}
    </div>
  );
};

export default AssessmentOverview;
