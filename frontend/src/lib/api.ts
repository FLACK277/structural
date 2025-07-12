
// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://structural-xjxr.onrender.com';

// API helper functions
export const api = {
  // Base URL for all API calls
  baseURL: API_BASE_URL,
  
  // Helper function to make API calls
  async fetch(endpoint: string, options: RequestInit = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Network error' }));
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  },
  
  // Auth helper
  async fetchWithAuth(endpoint: string, token: string, options: RequestInit = {}) {
    return this.fetch(endpoint, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${token}`,
      },
    });
  }
};

// Common API endpoints
export const endpoints = {
  login: '/api/login',
  register: '/api/register',
  me: '/api/me',
  health: '/health',
  roles: '/roles',
  assessmentQuestions: '/api/assessment_questions',
  submitAssessment: '/api/submit_assessment',
  userAssessments: '/api/user_assessments',
  recommendJobs: '/recommend_jobs',
  extractSkills: '/api/extract_skills',
  extractCertificateSkills: '/api/extract_certificate_skills',
  assessSkill: '/api/assess_skill',
  generateLearningPath: '/api/generate_learning_path',
  processProfile: '/api/process_profile',
  updateSkillProgress: '/api/update_skill_progress',
  dashboard: '/api/dashboard',
};