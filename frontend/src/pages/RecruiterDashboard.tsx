
import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Textarea } from '@/components/ui/textarea';
import { Users, TrendingUp, Clock, Search, Grid, List } from 'lucide-react';

const RecruiterDashboard = () => {
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchTerm, setSearchTerm] = useState('');

  const candidates = [
    {
      id: 1,
      name: 'Priya Sharma',
      education: 'BTech CSE, IIT Delhi',
      skills: ['React', 'Node.js', 'Python', 'AWS'],
      relevance: 95,
      location: 'Delhi',
      experience: '2 years'
    },
    {
      id: 2,
      name: 'Rahul Patel',
      education: 'MTech AI, BITS Pilani',
      skills: ['Machine Learning', 'TensorFlow', 'Python', 'SQL'],
      relevance: 89,
      location: 'Bangalore',
      experience: '3 years'
    },
    {
      id: 3,
      name: 'Anita Singh',
      education: 'BCA, Delhi University',
      skills: ['Java', 'Spring Boot', 'MySQL', 'Angular'],
      relevance: 87,
      location: 'Mumbai',
      experience: '1.5 years'
    },
    {
      id: 4,
      name: 'Vikram Kumar',
      education: 'BE Electronics, Anna University',
      skills: ['IoT', 'C++', 'Arduino', 'Python'],
      relevance: 82,
      location: 'Chennai',
      experience: '2.5 years'
    }
  ];

  const analytics = [
    { title: 'Total Applicants', value: '234', change: '+12%', icon: Users },
    { title: 'Best Skill Match', value: '95%', change: '+5%', icon: TrendingUp },
    { title: 'Time-to-Hire', value: '18 days', change: '-3 days', icon: Clock }
  ];

  const filteredCandidates = candidates.filter(candidate =>
    candidate.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    candidate.skills.some(skill => skill.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-6 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Recruiter Dashboard</h1>
            <p className="text-muted-foreground mt-1">Find and hire the best talent</p>
          </div>
          <Button className="bg-gradient-primary text-white hover:opacity-90">
            Post New Job
          </Button>
        </div>

        {/* Analytics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {analytics.map((item, index) => (
            <Card key={index} className="glass dark:glass-dark">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">{item.title}</p>
                    <p className="text-2xl font-bold text-foreground">{item.value}</p>
                    <p className="text-sm text-green-500">{item.change}</p>
                  </div>
                  <item.icon className="h-8 w-8 text-primary" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Candidate Browser */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="glass dark:glass-dark">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Candidate Browser</CardTitle>
                  <div className="flex items-center space-x-2">
                    <div className="relative">
                      <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search candidates..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="pl-10 w-64"
                      />
                    </div>
                    <Button
                      variant={viewMode === 'grid' ? 'default' : 'outline'}
                      size="icon"
                      onClick={() => setViewMode('grid')}
                    >
                      <Grid className="h-4 w-4" />
                    </Button>
                    <Button
                      variant={viewMode === 'list' ? 'default' : 'outline'}
                      size="icon"
                      onClick={() => setViewMode('list')}
                    >
                      <List className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className={viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 gap-4' : 'space-y-4'}>
                  {filteredCandidates.map((candidate) => (
                    <Card key={candidate.id} className="glass dark:glass-dark hover:scale-105 transition-transform">
                      <CardContent className="p-4">
                        <div className="flex items-start space-x-4">
                          <Avatar className="h-12 w-12">
                            <AvatarImage src={`/placeholder.svg`} />
                            <AvatarFallback>{candidate.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                          </Avatar>
                          <div className="flex-1 space-y-2">
                            <div className="flex items-center justify-between">
                              <h3 className="font-semibold text-foreground">{candidate.name}</h3>
                              <Badge variant="secondary" className="bg-green-100 text-green-800">
                                {candidate.relevance}% match
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground">{candidate.education}</p>
                            <p className="text-sm text-muted-foreground">{candidate.location} • {candidate.experience}</p>
                            <div className="flex flex-wrap gap-1">
                              {candidate.skills.slice(0, 3).map((skill) => (
                                <Badge key={skill} variant="outline" className="text-xs">
                                  {skill}
                                </Badge>
                              ))}
                              {candidate.skills.length > 3 && (
                                <Badge variant="outline" className="text-xs">
                                  +{candidate.skills.length - 3}
                                </Badge>
                              )}
                            </div>
                            <div className="flex space-x-2 pt-2">
                              <Button size="sm" variant="outline">View Profile</Button>
                              <Button size="sm" className="bg-gradient-primary text-white hover:opacity-90">
                                Invite to Apply
                              </Button>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Post Job Form */}
          <div className="space-y-6">
            <Card className="glass dark:glass-dark">
              <CardHeader>
                <CardTitle>Post a Job</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="jobTitle">Job Title</Label>
                  <Input id="jobTitle" placeholder="e.g. Senior Frontend Developer" />
                </div>
                <div>
                  <Label htmlFor="jobDescription">Description</Label>
                  <Textarea id="jobDescription" placeholder="Job description..." rows={4} />
                </div>
                <div>
                  <Label htmlFor="skills">Required Skills</Label>
                  <Input id="skills" placeholder="React, Node.js, TypeScript..." />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="salary">Salary (₹)</Label>
                    <Input id="salary" placeholder="8-12 LPA" />
                  </div>
                  <div>
                    <Label htmlFor="location">Location</Label>
                    <Input id="location" placeholder="Bangalore" />
                  </div>
                </div>
                <Button className="w-full bg-gradient-primary text-white hover:opacity-90">
                  Post Job
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecruiterDashboard;
