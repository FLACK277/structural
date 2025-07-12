from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
# Remove the top-level import and instantiation
# from app.models.skill.skill_matcher import SkillAssessmentEngine, save_assessment_result
import sqlite3
import json

router = APIRouter()

# Lazy loading for assessment engine
_assessment_engine = None

def get_assessment_engine():
    """Lazy load the assessment engine only when needed"""
    global _assessment_engine
    if _assessment_engine is None:
        from app.models.skill.skill_matcher import SkillAssessmentEngine, save_assessment_result
        _assessment_engine = SkillAssessmentEngine()
    return _assessment_engine

def save_assessment_result(user_id: str, results: Dict):
    """Lazy load the save function"""
    from app.models.skill.skill_matcher import save_assessment_result
    return save_assessment_result(user_id, results)

class AssessmentQuestionOut(BaseModel):
    question_id: int
    question: str
    options: List[str]
    difficulty: str
    concept: str

class AssessmentAnswerIn(BaseModel):
    question_id: int
    selected_option: int

class SubmitAssessmentRequest(BaseModel):
    user_id: str
    skill: str
    answers: List[AssessmentAnswerIn]

@router.get('/api/assessment_questions', response_model=List[AssessmentQuestionOut])
async def get_assessment_questions(skill: str = Query(...), num_questions: int = Query(3)):
    try:
        # Lazy load the assessment engine
        assessment_engine = get_assessment_engine()
        questions = assessment_engine.assessment_questions.get(skill)
        if not questions:
            raise HTTPException(status_code=404, detail="No questions found for this skill.")
        # Return only the requested number, and do not include the correct answer
        result = [
            AssessmentQuestionOut(
                question_id=i,
                question=q['question'],
                options=q['options'],
                difficulty=q['difficulty'],
                concept=q['concept']
            )
            for i, q in enumerate(questions[:num_questions])
        ]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/api/submit_assessment')
async def submit_assessment(request: SubmitAssessmentRequest):
    try:
        # Lazy load the assessment engine
        assessment_engine = get_assessment_engine()
        questions = assessment_engine.assessment_questions.get(request.skill)
        if not questions:
            raise HTTPException(status_code=404, detail="No questions found for this skill.")
        # Evaluate answers
        user_responses = []
        for ans in request.answers:
            if ans.question_id >= len(questions):
                continue
            q = questions[ans.question_id]
            is_correct = (ans.selected_option == q['correct'])
            user_responses.append({
                'question_id': ans.question_id,
                'response': ans.selected_option,
                'correct': is_correct,
                'difficulty': q['difficulty'],
                'concept': q['concept']
            })
        # Calculate results using the same logic as the engine
        assessment_results = {
            'user_id': request.user_id,
            'skill': request.skill,
            'questions_answered': len(user_responses),
            'correct_answers': sum(1 for r in user_responses if r['correct']),
            'estimated_level': assessment_engine._calculate_skill_level(user_responses),
            'confidence': assessment_engine._calculate_confidence(user_responses),
            'areas_for_improvement': assessment_engine._identify_weak_areas(user_responses),
            'strong_areas': assessment_engine._identify_strong_areas(user_responses)
        }
        save_assessment_result(request.user_id, assessment_results)
        return assessment_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/api/user_assessments/{user_id}')
async def get_user_assessments(user_id: str):
    conn = sqlite3.connect('learning_system.db')
    cursor = conn.cursor()
    cursor.execute('SELECT skill, score, level, confidence, assessment_data, created_at FROM assessments WHERE user_id = ?', (user_id,))
    rows = cursor.fetchall()
    conn.close()
    # Parse assessment_data JSON for each row
    assessments = []
    for skill, score, level, confidence, assessment_data, created_at in rows:
        try:
            data = json.loads(assessment_data) if assessment_data else {}
        except Exception:
            data = {}
        assessments.append({
            'skill': skill,
            'score': score,
            'level': level,
            'confidence': confidence,
            'created_at': created_at,
            **data
        })
    return assessments 