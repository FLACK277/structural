import { useEffect, useState } from 'react';
import { useAuth } from '@/lib/AuthContext';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { MapPin, Briefcase, IndianRupee, Star } from 'lucide-react';
import { api, endpoints } from '@/lib/api'; // Add this import

// Add type for job recommendation
interface JobRecommendation {
  job_id?: number;
  job_title?: string;
  industry?: string;
  functional_area?: string;
  experience_required?: string;
  key_skills?: string;
  salary?: string;
  rank?: number;
  company?: string;
}

type JobRecommendationsProps = { dashboardData?: any };

const JobRecommendations = ({ dashboardData }: JobRecommendationsProps) => {
  const { token } = useAuth();
  const [jobs, setJobs] = useState<JobRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  console.log('dashboardData:', dashboardData);

  const skillsArr = dashboardData?.top_skills?.length
    ? dashboardData.top_skills
    : dashboardData?.resume_skills || [];

  useEffect(() => {
    const fetchJobs = async () => {
      setLoading(true);
      setError('');
      console.log('Fetching jobs...');
      console.log('About to fetch jobs with:', {
        skills: skillsArr.map((s: any) => s.name).join(','),
        experience: dashboardData?.experience || 0,
        role_category: dashboardData?.role_category || '',
        industry: dashboardData?.industry || '',
        functional_area: dashboardData?.functional_area || '',
        job_title: dashboardData?.job_title || '',
        expected_salary: dashboardData?.expected_salary || 0,
      });
      try {
        const data = await api.fetchWithAuth(endpoints.recommendJobs, token, {
          method: 'POST',
          body: JSON.stringify({
            skills: skillsArr.map((s: any) => s.name).join(','),
            experience: dashboardData?.experience || 0,
            role_category: dashboardData?.role_category || '',
            industry: dashboardData?.industry || '',
            functional_area: dashboardData?.functional_area || '',
            job_title: dashboardData?.job_title || '',
            expected_salary: dashboardData?.expected_salary || 0,
          }),
        });
        console.log('Job recommendations API response:', data);
        if (data.recommendations) {
          setJobs(data.recommendations);
        } else {
          setJobs([]);
        }
      } catch (err) {
        console.error('Job fetch error:', err);
        setError(err instanceof Error ? err.message : 'Could not load job recommendations');
        setJobs([]);
      } finally {
        setLoading(false);
      }
    };
    if (dashboardData && skillsArr.length > 0) fetchJobs();
  }, [dashboardData, token]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold">Job Recommendations</h2>
        <Button variant="outline">Filter Jobs</Button>
      </div>
      
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Matches', value: jobs.length.toString(), icon: '🎯' },
          { label: 'Applied', value: '12', icon: '📝' },
          { label: 'In Review', value: '5', icon: '👀' },
          { label: 'Interviews', value: '2', icon: '💼' },
        ].map((stat, index) => (
          <Card key={index} className="glass dark:glass-dark">
            <CardContent className="p-4 text-center">
              <div className="text-2xl mb-2">{stat.icon}</div>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-sm text-muted-foreground">{stat.label}</p>
            </CardContent>
          </Card>
        ))}
      </div>
      
      {/* Job List */}
      {loading ? (
        <div>Loading job recommendations...</div>
      ) : error ? (
        <div className="text-red-500">{error}</div>
      ) : jobs.length === 0 ? (
        <div className="text-muted-foreground">No job recommendations found.</div>
      ) : (
        <div className="space-y-4">
          {jobs.map((job, idx) => (
            <Card
              key={job.job_id || idx}
              className={`glass dark:glass-dark transition-all hover:scale-[1.02] ${idx === 0 ? 'ring-2 ring-primary/20 bg-primary/5' : ''}`}
            >
              <CardContent className="p-6">
                <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h3 className="text-xl font-semibold">{job.job_title || 'N/A'}</h3>
                      {idx === 0 && (
                        <Badge className="bg-gradient-primary text-white">
                          <Star className="h-3 w-3 mr-1" />
                          Top Match
                        </Badge>
                      )}
                    </div>
                    {/* Show company if available */}
                    {job.company && (
                      <p className="text-lg text-muted-foreground mb-1">{job.company}</p>
                    )}
                    <p className="text-lg text-muted-foreground mb-1">{job.industry || 'Industry N/A'}</p>
                    <div className="flex flex-wrap gap-4 text-sm text-muted-foreground mb-3">
                      <div className="flex items-center">
                        <MapPin className="h-4 w-4 mr-1" />
                        {job.functional_area || 'N/A'}
                      </div>
                      <div className="flex items-center">
                        <IndianRupee className="h-4 w-4 mr-1" />
                        {job.salary || 'N/A'}
                      </div>
                      <div className="flex items-center">
                        <Briefcase className="h-4 w-4 mr-1" />
                        {job.experience_required || 'N/A'}
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {String(job.key_skills || '')
                        .split(',')
                        .filter((skill) => skill.trim().length > 0)
                        .map((skill: string) => (
                          <Badge key={skill.trim()} variant="secondary">
                            {skill.trim()}
                          </Badge>
                        ))}
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-3">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-primary">#{job.rank || idx + 1}</div>
                      <p className="text-sm text-muted-foreground">Rank</p>
                    </div>
                    <div className="flex gap-2">
                      {/* <Button variant="outline" size="sm">Save</Button>
                      <Button className="bg-gradient-primary" size="sm">Apply</Button> */}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default JobRecommendations;