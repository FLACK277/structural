from pydantic import BaseModel
from typing import List, Optional

class AssessmentStartRequest(BaseModel):
    name: str
    age: int
    role: str
    experience_level: str

class Answer(BaseModel):
    question_id: str
    answer: Optional[str] = None
    code: Optional[str] = None

class SubmitRequest(BaseModel):
    assessment_id: str
    user_answers: List[Answer]
