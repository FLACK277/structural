import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Edit, MapPin, Briefcase, GraduationCap, Plus, Trash2 } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { useAuth } from '@/lib/AuthContext';

interface MyProfileProps {
  dashboardData?: any;
  refreshDashboard?: () => void;
  userId?: string;
}

const MyProfile = ({ dashboardData, refreshDashboard, userId }: MyProfileProps) => {
  const safeDashboardData = dashboardData || {};
  // Extract user info from dashboardData
  const name = safeDashboardData.name || '';
  const headline = safeDashboardData.headline || '';
  const location = safeDashboardData.location || '';
  const education = safeDashboardData.education || [];
  const topSkills = safeDashboardData.top_skills || [];
  const resumeSkills = safeDashboardData.resume_skills || safeDashboardData.extracted_skills || [];
  const skills = topSkills.length > 0 ? topSkills : resumeSkills;
  const [editing, setEditing] = useState(false);
  const [resumeFile, setResumeFile] = useState(null);
  const [certificateFiles, setCertificateFiles] = useState([]);
  const [targetSkills, setTargetSkills] = useState(Array.isArray(skills) ? skills.map((s: any) => s.name || s).join(', ') : '');
  const [formLoading, setFormLoading] = useState(false);
  const [formError, setFormError] = useState('');
  const [formSuccess, setFormSuccess] = useState('');
  const { token } = useAuth();

  // Career goals state for editing
  const initialCareerGoal = (safeDashboardData.career_goal || safeDashboardData.career_goals || '').toString();
  const [careerGoalEdit, setCareerGoalEdit] = useState(initialCareerGoal);
  const [careerGoalEditing, setCareerGoalEditing] = useState(false);

  // Education state for editing (use a different variable name to avoid redeclaration)
  const initialEducationEdit = (safeDashboardData.education && Array.isArray(safeDashboardData.education)) ? safeDashboardData.education : [
    { degree: '', school: '', years: '', cgpa: '' }
  ];
  const [educationEdit, setEducationEdit] = useState(initialEducationEdit);
  const [nameEdit, setNameEdit] = useState(name);
  const [headlineEdit, setHeadlineEdit] = useState(headline);
  const [locationEdit, setLocationEdit] = useState(location);

  // When entering edit mode, pre-fill all fields with current values
  const handleEditClick = () => {
    setNameEdit(name);
    setHeadlineEdit(headline);
    setLocationEdit(location);
    setEducationEdit(initialEducationEdit);
    setTargetSkills(Array.isArray(skills) ? skills.map((s: any) => s.name || s).join(', ') : '');
    setCareerGoalEdit(initialCareerGoal);
    setEditing(true);
    setFormError('');
    setFormSuccess('');
  };

  const handleCancel = () => {
    setEditing(false);
    setFormError('');
    setFormSuccess('');
    setCareerGoalEditing(false);
  };

  const handleEducationChange = (idx, field, value) => {
    setEducationEdit(education => education.map((ed, i) => i === idx ? { ...ed, [field]: value } : ed));
  };
  const handleAddEducation = () => {
    setEducationEdit([...educationEdit, { degree: '', school: '', years: '', cgpa: '' }]);
  };
  const handleRemoveEducation = (idx) => {
    setEducationEdit(education => education.filter((_, i) => i !== idx));
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setFormLoading(true);
    setFormError('');
    setFormSuccess('');
    try {
      const formData = new FormData();
      formData.append('user_id', userId || '');
      if (resumeFile) formData.append('resume', resumeFile);
      for (let i = 0; i < certificateFiles.length; i++) {
        formData.append('certificates', certificateFiles[i]);
      }
      // Split target skills by comma and trim
      const skillsArr = targetSkills.split(',').map(s => s.trim()).filter(Boolean);
      for (let skill of skillsArr) {
        formData.append('target_skills', skill);
      }
      // Add education as JSON string
      formData.append('education', JSON.stringify(educationEdit));
      // Add new fields
      formData.append('name', nameEdit);
      formData.append('headline', headlineEdit);
      formData.append('location', locationEdit);
      formData.append('career_goal', careerGoalEdit);
      const res = await fetch('http://localhost:8000/api/process_profile', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });
      if (!res.ok) throw new Error('Failed to update profile');
      setFormSuccess('Profile updated!');
      setResumeFile(null);
      setCertificateFiles([]);
      setEditing(false);
      setCareerGoalEditing(false);
      if (refreshDashboard) refreshDashboard();
    } catch (err) {
      setFormError('Could not update profile.');
    } finally {
      setFormLoading(false);
    }
  };

  if (editing) {
    return (
      <div className="space-y-6">
        <h2 className="text-3xl font-bold mb-2">Edit Profile</h2>
        {formError && (
          <Alert variant="destructive" className="mb-2">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{formError}</AlertDescription>
          </Alert>
        )}
        {formSuccess && (
          <Alert className="mb-2">
            <AlertTitle>Success</AlertTitle>
            <AlertDescription>{formSuccess}</AlertDescription>
          </Alert>
        )}
        <form onSubmit={handleFormSubmit} className="space-y-4">
          <div>
            <Label htmlFor="name">Full Name</Label>
            <Input id="name" type="text" value={nameEdit} onChange={e => setNameEdit(e.target.value)} placeholder="Your full name" />
          </div>
          <div>
            <Label htmlFor="headline">Headline</Label>
            <Input id="headline" type="text" value={headlineEdit} onChange={e => setHeadlineEdit(e.target.value)} placeholder="e.g. Computer Science Student" />
          </div>
          <div>
            <Label htmlFor="location">Location</Label>
            <Input id="location" type="text" value={locationEdit} onChange={e => setLocationEdit(e.target.value)} placeholder="e.g. Delhi, India" />
          </div>
          <div>
            <Label htmlFor="resume">Resume (PDF/DOCX)</Label>
            <Input id="resume" type="file" accept=".pdf,.doc,.docx" onChange={e => setResumeFile(e.target.files[0])} />
          </div>
          <div>
            <Label htmlFor="certificates">Certificates (optional, multiple)</Label>
            <Input id="certificates" type="file" multiple onChange={e => setCertificateFiles(Array.from(e.target.files))} />
          </div>
          <div>
            <Label htmlFor="target_skills">Target Skills (comma separated)</Label>
            <Input id="target_skills" type="text" value={targetSkills} onChange={e => setTargetSkills(e.target.value)} placeholder="e.g. Python, Data Analysis, React" />
          </div>
          <div>
            <Label>Education</Label>
            {educationEdit.map((ed, idx) => (
              <div key={idx} className="flex flex-wrap gap-2 mb-2 items-end">
                <Input
                  placeholder="Degree"
                  value={ed.degree}
                  onChange={e => handleEducationChange(idx, 'degree', e.target.value)}
                  className="w-32"
                />
                <Input
                  placeholder="School"
                  value={ed.school}
                  onChange={e => handleEducationChange(idx, 'school', e.target.value)}
                  className="w-48"
                />
                <Input
                  placeholder="Years"
                  value={ed.years}
                  onChange={e => handleEducationChange(idx, 'years', e.target.value)}
                  className="w-28"
                />
                <Input
                  placeholder="CGPA"
                  value={ed.cgpa}
                  onChange={e => handleEducationChange(idx, 'cgpa', e.target.value)}
                  className="w-20"
                />
                <Button type="button" variant="ghost" size="icon" onClick={() => handleRemoveEducation(idx)} disabled={educationEdit.length === 1}>
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
            <Button type="button" variant="outline" size="sm" onClick={handleAddEducation} className="mt-2">
              <Plus className="h-4 w-4 mr-1" /> Add Education
            </Button>
          </div>
          <div className="flex gap-2">
            <Button type="submit" className="w-full" disabled={formLoading}>
              {formLoading ? 'Saving...' : 'Save'}
            </Button>
            <Button type="button" variant="outline" onClick={handleCancel} disabled={formLoading}>
              Cancel
            </Button>
          </div>
        </form>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold">My Profile</h2>
        <Button className="bg-gradient-primary" onClick={handleEditClick}>
          <Edit className="h-4 w-4 mr-2" />
          Edit Profile
        </Button>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Profile Card */}
        <Card className="lg:col-span-1 glass dark:glass-dark">
          <CardContent className="p-6 text-center">
            <Avatar className="h-24 w-24 mx-auto mb-4">
              <AvatarImage src="/placeholder.svg" />
              <AvatarFallback className="text-2xl">{name ? name[0] : 'U'}</AvatarFallback>
            </Avatar>
            <h3 className="text-xl font-semibold mb-2">{name || 'Your Name'}</h3>
            <p className="text-muted-foreground mb-4">{headline || 'Add a headline'}</p>
            <div className="space-y-3 text-left">
              {education.length > 0 && education[0]?.degree && (
                <div className="flex items-center gap-2 text-sm">
                <GraduationCap className="h-4 w-4 text-primary" />
                  <span>{education[0].degree}{education[0].school ? `, ${education[0].school}` : ''}</span>
              </div>
              )}
              {location && (
                <div className="flex items-center gap-2 text-sm">
                <MapPin className="h-4 w-4 text-primary" />
                  <span>{location}</span>
              </div>
              )}
            </div>
          </CardContent>
        </Card>
        {/* Education & Skills */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <Card className="mb-4">
            <CardHeader>
              <CardTitle>Education</CardTitle>
            </CardHeader>
            <CardContent>
              {education.length > 0 ? (
                <ul className="space-y-2">
                  {education.map((ed, idx) => (
                    <li key={idx} className="flex flex-col md:flex-row md:items-center md:gap-4">
                      <span className="font-medium">{ed.degree}</span>
                      <span className="text-muted-foreground">{ed.school || ed.institution}</span>
                      <span className="text-muted-foreground">{ed.years || ed.year}</span>
                      {ed.cgpa && <span className="text-muted-foreground">CGPA: {ed.cgpa}</span>}
                    </li>
                  ))}
                </ul>
              ) : (
                <span className="text-muted-foreground">No education info added.</span>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Key Skills</CardTitle>
            </CardHeader>
            <CardContent>
              {skills.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                  {skills.map((skill, idx) => (
                    <Badge key={idx} variant="secondary">{skill.name || skill}</Badge>
                ))}
              </div>
              ) : (
                <span className="text-muted-foreground">No skills added.</span>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Career Goals</CardTitle>
            </CardHeader>
            <CardContent>
              {careerGoalEditing ? (
                <form onSubmit={handleFormSubmit} className="flex gap-2 items-center">
                  <Input
                    type="text"
                    value={careerGoalEdit}
                    onChange={e => setCareerGoalEdit(e.target.value)}
                    placeholder="Enter your career goal(s)"
                    className="w-64"
                  />
                  <Button type="submit" size="sm" disabled={formLoading}>
                    {formLoading ? 'Saving...' : 'Save'}
                  </Button>
                  <Button type="button" variant="outline" size="sm" onClick={handleCancel} disabled={formLoading}>
                    Cancel
                  </Button>
                </form>
              ) : careerGoalEdit ? (
                <div className="flex items-center gap-2">
                  <span>{careerGoalEdit}</span>
                  <Button type="button" size="sm" variant="ghost" onClick={() => setCareerGoalEditing(true)}>
                    Edit
                  </Button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <span className="text-muted-foreground">No career goal set.</span>
                  <Button type="button" size="sm" variant="ghost" onClick={() => setCareerGoalEditing(true)}>
                    Add
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default MyProfile;
