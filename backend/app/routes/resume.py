from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import io
import json
from app.models.skill.skill_matcher import PersonalizedLearningSystem, Skill, LearningResource, LearningPath
from pydantic import BaseModel
import os
import uuid
import cv2
import numpy as np
from PIL import Image
import bcrypt
import jwt
from datetime import datetime, timedelta
import sqlite3
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security

UPLOAD_DIR = "uploads"

router = APIRouter()

# Initialize the BERT-enhanced system (singleton)
bert_learning_system = PersonalizedLearningSystem()

JWT_SECRET = "your_super_secret_key"  # Change this in production
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

# --- Pydantic Schemas ---
class SkillIn(BaseModel):
    name: str
    level: float
    category: Optional[str] = 'general'
    confidence: Optional[float] = 0.7
    source: Optional[str] = 'user_input'

class LearningResourceOut(BaseModel):
    title: str
    description: str
    difficulty: str
    duration: int
    skills: List[str]
    url: str
    rating: float

class LearningPathOut(BaseModel):
    skills_gap: List[str]
    recommended_resources: List[LearningResourceOut]
    estimated_completion: int
    priority_order: List[str]

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    user_id: str
    username: str
    email: str

# --- API Endpoints ---

@router.post('/api/extract_skills')
async def api_extract_skills(resume: UploadFile = File(...)):
    try:
        # Ensure the uploads directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        # Save the uploaded file
        file_location = os.path.join(UPLOAD_DIR, resume.filename)
        with open(file_location, "wb") as f:
            f.write(await resume.read())

        # Now extract text from the saved file
        resume_text = bert_learning_system.resume_parser.extract_text_from_resume(file_location)
        print("Extracted text:", resume_text[:500])
        skills = bert_learning_system.resume_parser.extract_skills(resume_text)
        print("Extracted skills:", skills)
        return {"skills": [skill.__dict__ for skill in skills]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/api/extract_certificate_skills')
async def api_extract_certificate_skills(certificate: UploadFile = File(...)):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        unique_filename = f"{uuid.uuid4()}_{certificate.filename}"
        file_location = os.path.join(UPLOAD_DIR, unique_filename)
        with open(file_location, "wb") as f:
            f.write(await certificate.read())

        skills = bert_learning_system.cv_extractor.extract_from_certificate(file_location)
        print("Extracted skills:", skills)
        return {"skills": [skill.__dict__ for skill in skills]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AssessmentRequest(BaseModel):
    user_id: str
    skill: str
    num_questions: Optional[int] = 10

@router.post('/api/assess_skill')
async def api_assess_skill(request: AssessmentRequest):
    try:
        result = bert_learning_system.assessment_engine.conduct_assessment(
            request.user_id, request.skill, request.num_questions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LearningPathRequest(BaseModel):
    user_id: str
    current_skills: List[SkillIn]
    target_skills: List[str]
    preferences: Optional[Dict[str, Any]] = None

@router.post('/api/generate_learning_path')
async def api_generate_learning_path(request: LearningPathRequest):
    try:
        current_skills = [Skill(**s.dict()) for s in request.current_skills]
        path = bert_learning_system.path_generator.generate_learning_path(
            current_skills, request.target_skills, request.preferences)
        return {
            'skills_gap': path.skills_gap,
            'recommended_resources': [r.__dict__ for r in path.recommended_resources],
            'estimated_completion': path.estimated_completion,
            'priority_order': path.priority_order
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/api/process_profile')
async def api_process_profile(
    user_id: str = Form(...),
    target_skills: List[str] = Form([]),
    resume: Optional[UploadFile] = File(None),
    certificates: Optional[List[UploadFile]] = File(None),
    name: str = Form(""),
    headline: str = Form(""),
    education: str = Form("[]"),
    location: str = Form(""),
    career_goal: str = Form("")
):
    try:
        cert_files = certificates or []
        resume_path = None
        if resume:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            resume_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{resume.filename}")
            with open(resume_path, "wb") as f:
                f.write(await resume.read())

        certificate_paths = []
        for c in cert_files:
            cert_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{c.filename}")
            with open(cert_path, "wb") as f:
                f.write(await c.read())
            certificate_paths.append(cert_path)

        # Parse education JSON string
        try:
            education_data = json.loads(education)
        except Exception:
            education_data = []

        result = bert_learning_system.process_user_profile(
            user_id=user_id,
            resume_path=resume_path,
            certificate_paths=certificate_paths,
            target_skills=target_skills,
            name=name,
            headline=headline,
            education=education_data,
            location=location,
            career_goal=career_goal
        )
        return result
    except Exception as e:
        print("Error in /api/process_profile:", e)  # <--- Add this line
        raise HTTPException(status_code=500, detail=str(e))

class SkillProgressRequest(BaseModel):
    user_id: str
    skill_name: str
    new_level: float
    completion_data: Optional[Dict[str, Any]] = None

@router.post('/api/update_skill_progress')
async def api_update_skill_progress(request: SkillProgressRequest):
    try:
        result = bert_learning_system.update_skill_progress(
            request.user_id, request.skill_name, request.new_level, request.completion_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/api/dashboard/{user_id}')
async def api_dashboard(user_id: str):
    try:
        data = bert_learning_system.get_dashboard_data(user_id)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Auth Endpoints ---

@router.post('/api/register')
async def register_user(request: RegisterRequest):
    try:
        conn = sqlite3.connect('learning_system.db')
        cursor = conn.cursor()
        # Check if username or email exists
        cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (request.username, request.email))
        if cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=400, detail="Username or email already exists.")
        # Hash password
        hashed_pw = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt())
        user_id = str(uuid.uuid4())
        cursor.execute('''INSERT INTO users (user_id, username, email, password_hash) VALUES (?, ?, ?, ?)''',
                       (user_id, request.username, request.email, hashed_pw.decode('utf-8')))
        conn.commit()
        conn.close()
        return {"message": "Registration successful!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/api/login')
async def login_user(request: LoginRequest):
    try:
        conn = sqlite3.connect('learning_system.db')
        cursor = conn.cursor()
        cursor.execute('SELECT user_id, username, email, password_hash FROM users WHERE username = ?', (request.username,))
        user = cursor.fetchone()
        conn.close()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        user_id, username, email, password_hash = user
        if not bcrypt.checkpw(request.password.encode('utf-8'), password_hash.encode('utf-8')):
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        payload = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "exp": datetime.utcnow() + timedelta(seconds=JWT_EXP_DELTA_SECONDS)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return {"access_token": token, "token_type": "bearer"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

@router.get('/api/me', response_model=UserOut)
async def get_me(user=Depends(get_current_user)):
    return UserOut(user_id=user["user_id"], username=user["username"], email=user["email"])
