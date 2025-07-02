import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { FileText, GraduationCap, Target, Award } from 'lucide-react';
import { useMemo } from 'react';

interface ResumeInsightsProps {
  education?: { degree: string; institution?: string; school?: string; year?: string; years?: string; cgpa?: string }[];
  resume_skills?: { name: string; confidence: number }[];
  career_goals?: string[];
  profile_completeness?: number;
}

const ResumeInsights = ({ education = [], resume_skills = [], career_goals = [], profile_completeness = 0 }: ResumeInsightsProps) => {
  // Normalize education fields
  const normalizedEducation = useMemo(() =>
    (education || []).map(ed => ({
      degree: ed.degree || '',
      institution: ed.institution || ed.school || '',
      year: ed.year || ed.years || '',
      cgpa: ed.cgpa || ''
    })), [education]);

  return (
    <Card className="glass dark:glass-dark border-border/20">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <FileText className="h-5 w-5 text-primary" />
          <span>Resume Insights</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Resume Completeness */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium">Profile Completeness</span>
            <span className="text-sm text-muted-foreground">{profile_completeness}%</span>
          </div>
          <Progress value={profile_completeness} className="h-2" />
        </div>

        {/* Education */}
        {normalizedEducation.length > 0 && (
          <div>
            <h4 className="flex items-center space-x-2 text-sm font-medium mb-3">
              <GraduationCap className="h-4 w-4 text-primary" />
              <span>Education</span>
            </h4>
            <div className="space-y-2">
              {normalizedEducation.map((edu, index) => (
                <div key={index} className="flex justify-between items-center p-3 bg-muted/50 dark:bg-muted/20 rounded-lg">
                  <div>
                    <p className="font-medium text-sm">{edu.degree}</p>
                    <p className="text-xs text-muted-foreground">{edu.institution}</p>
                  </div>
                  <Badge variant="secondary">{edu.year}{edu.cgpa ? ` | CGPA: ${edu.cgpa}` : ''}</Badge>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Skills Analysis */}
        {resume_skills.length > 0 && (
          <div>
            <h4 className="flex items-center space-x-2 text-sm font-medium mb-3">
              <Award className="h-4 w-4 text-primary" />
              <span>Skill Strengths</span>
            </h4>
            <div className="space-y-3">
              {resume_skills.map((skill, index) => (
                <div key={index}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-sm font-medium">{skill.name}</span>
                    <span className="text-xs text-muted-foreground">{skill.confidence}%</span>
                  </div>
                  <Progress value={skill.confidence} className="h-2" />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Career Goals */}
        {career_goals.length > 0 && (
          <div>
            <h4 className="flex items-center space-x-2 text-sm font-medium mb-3">
              <Target className="h-4 w-4 text-primary" />
              <span>Career Goals</span>
            </h4>
            <div className="flex flex-wrap gap-2">
              {career_goals.map((goal, index) => (
                <Badge key={index} variant="outline" className="bg-primary/10 text-primary border-primary/20">
                  {goal}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* TODO: Backend Integration Point */}
        {/* Add API integration for:
            - Resume parsing results
            - Skill extraction confidence scores
            - Goal recommendations based on profile
            - Completion suggestions
        */}
      </CardContent>
    </Card>
  );
};

export default ResumeInsights;
