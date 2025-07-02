
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { TrendingUp, DollarSign, Target, Users } from 'lucide-react';

const CareerInsights = () => {
  const trendingSkills = [
    { skill: 'React', growth: '+25%', demand: 'High' },
    { skill: 'Python', growth: '+18%', demand: 'Very High' },
    { skill: 'AWS', growth: '+30%', demand: 'High' },
    { skill: 'Machine Learning', growth: '+40%', demand: 'Very High' },
  ];

  const salaryData = [
    { role: 'Frontend Developer', range: '6-15 LPA', avg: '10.5 LPA' },
    { role: 'Backend Developer', range: '8-18 LPA', avg: '12 LPA' },
    { role: 'Full Stack Developer', range: '10-22 LPA', avg: '15 LPA' },
    { role: 'Data Scientist', range: '12-25 LPA', avg: '18 LPA' },
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold">Career Insights</h2>
      
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { 
            title: 'Market Demand', 
            value: 'Growing', 
            subtitle: 'Tech sector +15%',
            icon: TrendingUp,
            color: 'text-green-500'
          },
          { 
            title: 'Avg Salary', 
            value: '₹12.5 LPA', 
            subtitle: 'For your skills',
            icon: DollarSign,
            color: 'text-blue-500'
          },
          { 
            title: 'Skill Gap', 
            value: '25%', 
            subtitle: 'Improvement needed',
            icon: Target,
            color: 'text-orange-500'
          },
          { 
            title: 'Competition', 
            value: 'Medium', 
            subtitle: '1.2M candidates',
            icon: Users,
            color: 'text-purple-500'
          },
        ].map((metric, index) => (
          <Card key={index} className="glass dark:glass-dark">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <metric.icon className={`h-5 w-5 ${metric.color}`} />
                <span className="text-xs text-muted-foreground">{metric.subtitle}</span>
              </div>
              <div className="text-xl font-bold">{metric.value}</div>
              <p className="text-sm text-muted-foreground">{metric.title}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trending Skills */}
        <Card className="glass dark:glass-dark">
          <CardHeader>
            <CardTitle>Trending Skills in Tech</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {trendingSkills.map((item, index) => (
                <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                  <div>
                    <h4 className="font-medium">{item.skill}</h4>
                    <p className="text-sm text-muted-foreground">Demand: {item.demand}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-green-500 font-semibold">{item.growth}</div>
                    <p className="text-xs text-muted-foreground">Growth</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Salary Insights */}
        <Card className="glass dark:glass-dark">
          <CardHeader>
            <CardTitle>Salary Ranges by Role</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {salaryData.map((item, index) => (
                <div key={index} className="p-3 rounded-lg bg-muted/50">
                  <div className="flex justify-between items-start mb-1">
                    <h4 className="font-medium">{item.role}</h4>
                    <span className="text-primary font-semibold">{item.avg}</span>
                  </div>
                  <p className="text-sm text-muted-foreground">Range: {item.range}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Market Insights */}
      <Card className="glass dark:glass-dark">
        <CardHeader>
          <CardTitle>Market Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">Hot Industries</h4>
              <div className="space-y-2">
                {['Fintech', 'E-commerce', 'EdTech', 'HealthTech'].map((industry) => (
                  <div key={industry} className="flex justify-between">
                    <span>{industry}</span>
                    <span className="text-green-500">↗</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-3">Hiring Trends</h4>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p>• Remote work opportunities increasing by 40%</p>
                <p>• Startups offering 20% higher packages</p>
                <p>• Focus on full-stack capabilities</p>
                <p>• AI/ML skills in high demand</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default CareerInsights;
