from fastapi import APIRouter, HTTPException
from app.models.jobrecommendation.job_recommendation_schema import JobRecommendationRequest
from app.models.jobrecommendation.recommendation_engine import BERTJobRecommenderSystem
import os


router = APIRouter()

# Load the precomputed BERTJobRecommenderSystem model from pickle
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/jobrecommendation/bert_job_recommender.pkl'))
recommender = None
try:
    recommender = BERTJobRecommenderSystem.load_model(MODEL_PATH)
except Exception as e:
    print(f"[Job Recommendation] Failed to load model: {e}")

@router.post("/recommend_jobs")
def recommend_jobs(request: JobRecommendationRequest):
    """
    Recommend jobs based on user profile input.
    """
    if recommender is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    user_profile = {
        'skills': request.skills.split(',') if isinstance(request.skills, str) else request.skills,
        'experience': request.experience,
        'role_category': request.role_category,
        'industry': request.industry,
        'functional_area': request.functional_area,
        'job_title': request.job_title,
        'expected_salary': request.expected_salary or 0
    }
    try:
        recommendations_df = recommender.recommend_jobs_for_user_profile(user_profile, top_n=5)
        recommendations = []
        for idx, row in recommendations_df.iterrows():
            recommendations.append({
                'job_id': int(row['Job_ID']),
                'job_title': str(row.get('Job Title', 'N/A')),
                'industry': str(row.get('Industry', 'N/A')),
                'functional_area': str(row.get('Functional Area', 'N/A')),
                'experience_required': str(row.get('Job Experience Required', 'N/A')),
                'key_skills': str(row.get('Key Skills', 'N/A')),
                'salary': str(row.get('Job Salary', 'N/A')),
                'rank': idx + 1
            })
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {e}")
