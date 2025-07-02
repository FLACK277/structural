from pydantic import BaseModel
from typing import Optional

class JobRecommendationRequest(BaseModel):
    skills: str
    experience: float
    role_category: str
    industry: str
    functional_area: str
    job_title: str
    expected_salary: Optional[float] = 0