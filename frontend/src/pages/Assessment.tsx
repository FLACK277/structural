import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/lib/AuthContext';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { api, endpoints } from '@/lib/api'; // Add this import

const staticSkills = [
  'python', 'machine_learning', 'javascript', 'sql', 'data_science', 'cloud_computing', 'react', 'aws', 'tensorflow', 'advanced_sql'
];

const Assessment = () => {
  const { token, user } = useAuth();
  const userId = user?.user_id;
  const navigate = useNavigate();
  const [step, setStep] = useState<'select' | 'quiz' | 'result'>('select');
  const [selectedSkill, setSelectedSkill] = useState('');
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  console.log('user:', user);
  console.log('userId:', userId);

  // 1. Skill selection
  const handleSkillSelect = async (skill: string) => {
    setSelectedSkill(skill);
    setStep('quiz');
    setLoading(true);
    setError('');
    try {
      const data = await api.fetchWithAuth(
        `${endpoints.assessmentQuestions}?skill=${encodeURIComponent(skill)}&num_questions=5`,
        token
      );
      setQuestions(data);
      setAnswers({});
    } catch (err) {
      setError('Could not load questions');
      setStep('select');
    } finally {
      setLoading(false);
    }
  };

  // 2. Answer selection
  const handleAnswer = (qid: number, option: number) => {
    setAnswers((prev) => ({ ...prev, [qid]: option }));
  };

  // 3. Submit assessment
  const handleSubmit = async () => {
    if (!userId) {
      setError('User not logged in');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const payload = {
        user_id: userId,
        skill: selectedSkill,
        answers: Object.entries(answers).map(([qid, selected_option]) => ({
          question_id: Number(qid),
          selected_option: Number(selected_option),
        })),
      };
      const data = await api.fetchWithAuth(endpoints.submitAssessment, token, {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      setResult(data);
      setStep('result');
    } catch (err) {
      setError('Could not submit assessment');
    } finally {
      setLoading(false);
    }
  };

  // 4. Render
  if (step === 'select') {
    return (
      <div className="max-w-xl mx-auto mt-12">
        <Card>
          <CardHeader>
            <CardTitle>Select a Skill to Assess</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              {staticSkills.map((skill) => (
                <Button key={skill} onClick={() => handleSkillSelect(skill)}>{skill.replace('_', ' ').toUpperCase()}</Button>
              ))}
            </div>
            {error && <div className="text-red-500 mt-4">{error}</div>}
          </CardContent>
        </Card>
      </div>
    );
  }

  if (step === 'quiz') {
    return (
      <div className="max-w-2xl mx-auto mt-12">
        <Card>
          <CardHeader>
            <CardTitle>Assessment: {selectedSkill.replace('_', ' ').toUpperCase()}</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? <div>Loading questions...</div> : (
              <form onSubmit={e => { e.preventDefault(); handleSubmit(); }}>
                <div className="space-y-6">
                  {questions.map((q: any, idx: number) => (
                    <div key={q.question_id} className="mb-4">
                      <div className="font-medium mb-2">Q{idx + 1}. {q.question}</div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {q.options.map((opt: string, i: number) => (
                          <Button
                            key={i}
                            type="button"
                            variant={answers[q.question_id] === i ? 'default' : 'outline'}
                            className="w-full"
                            onClick={() => handleAnswer(q.question_id, i)}
                          >
                            {opt}
                          </Button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                {error && <div className="text-red-500 mt-4">{error}</div>}
                <Button type="submit" className="mt-6 w-full" disabled={Object.keys(answers).length !== questions.length || loading}>
                  Submit Assessment
                </Button>
              </form>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  if (step === 'result') {
    return (
      <div className="max-w-xl mx-auto mt-12">
        <Card>
          <CardHeader>
            <CardTitle>Assessment Results</CardTitle>
          </CardHeader>
          <CardContent>
            {result ? (
              <div className="space-y-4">
                <div className="text-lg font-semibold">Skill: {selectedSkill.replace('_', ' ').toUpperCase()}</div>
                <div>Score: <span className="font-bold">{Math.round((result.estimated_level || 0) * 100)}%</span></div>
                <div>Confidence: <span className="font-bold">{Math.round((result.confidence || 0) * 100)}%</span></div>
                {result.areas_for_improvement && result.areas_for_improvement.length > 0 && (
                  <div>
                    <div className="font-medium">Areas for Improvement:</div>
                    <ul className="list-disc list-inside">
                      {result.areas_for_improvement.map((area: string, i: number) => (
                        <li key={i}>{area}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {result.strong_areas && result.strong_areas.length > 0 && (
                  <div>
                    <div className="font-medium">Strong Areas:</div>
                    <ul className="list-disc list-inside">
                      {result.strong_areas.map((area: string, i: number) => (
                        <li key={i}>{area}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <Button className="mt-6 w-full" onClick={() => navigate('/dashboard')}>Return to Dashboard</Button>
              </div>
            ) : <div>Could not load results.</div>}
          </CardContent>
        </Card>
      </div>
    );
  }

  return null;
};

export default Assessment;