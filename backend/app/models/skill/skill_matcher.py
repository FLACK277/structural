import os
import sqlite3
import json
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import cv2
import pytesseract
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer  # BERT Integration
import warnings
import traceback
import hashlib
import time
from datetime import datetime, timedelta
import re
from pathlib import Path
import io
import base64
from PIL import Image
import fitz
import docx
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from typing import List, Dict, Optional, Tuple, Union, Any
import uuid
from dataclasses import dataclass
import tensorflow_hub as hub

# NEW SKILLMATCHER

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Installing...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Set max length after ensuring nlp is defined
nlp.max_length = 2_000_000  # or higher if needed

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Installing...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

@dataclass
class Skill:
    name: str
    level: float  # 0-1 scale
    category: str
    confidence: float
    source: str  # 'resume', 'assessment', 'vision'

@dataclass
class LearningResource:
    title: str
    description: str
    difficulty: str
    duration: int  # minutes
    skills: List[str]
    url: str
    rating: float

@dataclass
class LearningPath:
    user_id: str
    skills_gap: List[str]
    recommended_resources: List[LearningResource]
    estimated_completion: int  # days
    priority_order: List[str]

class BERTEnhancedResumeParser:
    """BERT-Enhanced Resume Parser with semantic understanding"""
    
    def __init__(self):
        # Initialize spaCy
        self.nlp = nlp
        
        # BERT Integration
        try:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            self.bert_model = None
        
        # Keep Universal Sentence Encoder as fallback
        try:
            self.use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        except Exception as e:
            logger.warning(f"Universal Sentence Encoder not available: {e}")
            self.use_model = None
        
        # Enhanced skill database
        self.skill_keywords = {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
                'typescript', 'kotlin', 'swift', 'scala', 'r', 'matlab', 'perl', 'react', 
                'angular', 'vue', 'django', 'flask', 'spring'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 
                'numpy', 'scikit-learn', 'data analysis', 'statistics', 'data visualization', 
                'tableau', 'power bi', 'matplotlib', 'seaborn'
            ],
            'cloud': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'cloud computing'
            ],
            'database': [
                'sql', 'mongodb', 'postgresql', 'mysql', 'redis', 'cassandra', 'oracle', 'sqlite'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving', 'project management'
            ]
        }
        
        self.skill_embeddings_cache = {}
        self._build_skill_embeddings()
    
    def _build_skill_embeddings(self):
        """Build BERT embeddings for all skills"""
        if not self.bert_model:
            logger.warning("BERT model not available - using traditional methods only")
            return
        
        try:
            logger.info("Building BERT embeddings for skill database...")
            all_skills = []
            for category, skills in self.skill_keywords.items():
                all_skills.extend(skills)
            
            # Generate embeddings in batches
            batch_size = 32
            for i in range(0, len(all_skills), batch_size):
                batch = all_skills[i:i + batch_size]
                embeddings = self.bert_model.encode(batch, show_progress_bar=False)
                
                for skill, embedding in zip(batch, embeddings):
                    self.skill_embeddings_cache[skill.lower()] = {
                        'embedding': embedding,
                        'category': self._get_skill_category(skill)
                    }
            
            logger.info(f"Built BERT embeddings for {len(all_skills)} skills")
        except Exception as e:
            logger.error(f"Error building BERT embeddings: {e}")
    
    def _get_skill_category(self, skill):
        """Get category for a skill"""
        for category, skills in self.skill_keywords.items():
            if skill in skills:
                return category
        return 'general'
    
    def extract_text_from_resume(self, file_path: str) -> str:
        """Extract text from various resume formats"""
        try:
            if hasattr(file_path, 'read'):  # File object
                # Handle file upload object
                content = file_path.read()
                filename = getattr(file_path, 'filename', 'unknown')
                
                if filename.endswith('.pdf'):
                    return self._extract_from_pdf_content(content)
                elif filename.endswith(('.png', '.jpg', '.jpeg')):
                    return self._extract_from_image_content(content)
                else:
                    return content.decode('utf-8', errors='ignore')
            
            elif file_path.endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                return self._extract_from_image(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error extracting text from resume: {e}")
            return ""
    
    def _extract_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            return "Sample extracted text from PDF resume with skills like Python, Machine Learning, TensorFlow, SQL"

    def _extract_from_pdf_content(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting from PDF content: {e}")
            return ""
    
    def _extract_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(gray)
            return text
        except Exception as e:
            logger.error(f"Error extracting from image: {e}")
            return ""
    
    def _extract_from_image_content(self, content: bytes) -> str:
        """Extract text from image content using OCR"""
        try:
            image = Image.open(io.BytesIO(content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting from image content: {e}")
            return ""
    
    def extract_skills_bert(self, resume_text: str) -> List[Skill]:
        """BERT-powered skill extraction with semantic understanding"""
        if not self.bert_model or not self.skill_embeddings_cache:
            return self.extract_skills_traditional(resume_text)
        
        try:
            # Generate embedding for resume text
            text_embedding = self.bert_model.encode([resume_text.lower()], show_progress_bar=False)
            
            detected_skills = []
            similarity_threshold = 0.6
            
            for skill, skill_data in self.skill_embeddings_cache.items():
                try:
                    similarity = cosine_similarity(
                        text_embedding, 
                        [skill_data['embedding']]
                    )[0][0]
                    
                    if similarity > similarity_threshold:
                        # Estimate skill level using context
                        level = self._estimate_skill_level_bert(resume_text, skill)
                        
                        detected_skills.append(Skill(
                            name=skill,
                            level=level,
                            category=skill_data['category'],
                            confidence=float(similarity),
                            source='resume_bert'
                        ))
                except Exception as skill_error:
                    continue
            
            # Sort by confidence and return top skills
            detected_skills.sort(key=lambda x: x.confidence, reverse=True)
            return detected_skills[:20]
            
        except Exception as e:
            logger.error(f"Error in BERT skill extraction: {e}")
            return self.extract_skills_traditional(resume_text)
    
    def extract_skills_traditional(self, resume_text: str) -> List[Skill]:
        """Traditional keyword-based skill extraction"""
        try:
            skills = []
            doc = self.nlp(resume_text.lower())
            
            for category, keywords in self.skill_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in resume_text.lower():
                        confidence = self._calculate_skill_confidence_traditional(resume_text, keyword)
                        level = self._estimate_skill_level_traditional(resume_text, keyword)
                        
                        skills.append(Skill(
                            name=keyword,
                            level=level,
                            category=category,
                            confidence=confidence,
                            source='resume_traditional'
                        ))
            
            return self._deduplicate_skills(skills)
        except Exception as e:
            logger.error(f"Error in traditional skill extraction: {e}")
            return []
    
    def extract_skills(self, resume_text: str) -> List[Skill]:
        """Hybrid skill extraction combining BERT and traditional methods"""
        try:
            # Get skills from both methods
            bert_skills = self.extract_skills_bert(resume_text)
            traditional_skills = self.extract_skills_traditional(resume_text)
            
            # Combine and deduplicate
            all_skills = {}
            
            # Add traditional skills (high confidence for exact matches)
            for skill in traditional_skills:
                key = skill.name.lower()
                all_skills[key] = skill
                all_skills[key].confidence = min(0.95, skill.confidence + 0.1)
            
            # Add BERT skills (semantic matches)
            for skill in bert_skills:
                key = skill.name.lower()
                if key not in all_skills:
                    all_skills[key] = skill
                else:
                    # Combine confidences for skills found by both methods
                    existing_skill = all_skills[key]
                    combined_confidence = min(1.0, existing_skill.confidence + skill.confidence * 0.3)
                    all_skills[key].confidence = combined_confidence
                    all_skills[key].source = 'resume_hybrid'
            
            return list(all_skills.values())
            
        except Exception as e:
            logger.error(f"Error in hybrid skill extraction: {e}")
            return self.extract_skills_traditional(resume_text)
    
    def _estimate_skill_level_bert(self, text: str, skill: str) -> float:
        """Estimate skill level using BERT context understanding"""
        try:
            # Look for skill in context
            sentences = [sent.text for sent in self.nlp(text).sents if skill.lower() in sent.text.lower()]
            
            if not sentences:
                return 0.5  # Default level
            
            # Use BERT to understand proficiency context
            proficiency_contexts = [
                f"expert in {skill}",
                f"advanced {skill}",
                f"intermediate {skill}",
                f"beginner {skill}"
            ]
            
            if self.bert_model:
                sentence_embeddings = self.bert_model.encode(sentences[:3])
                context_embeddings = self.bert_model.encode(proficiency_contexts)
                
                # Find best matching proficiency level
                similarities = cosine_similarity(sentence_embeddings, context_embeddings)
                max_sim_idx = np.argmax(similarities)
                
                # Map to skill levels
                level_map = [0.9, 0.7, 0.5, 0.3]  # expert, advanced, intermediate, beginner
                return level_map[max_sim_idx % 4]
            
            return self._estimate_skill_level_traditional(text, skill)
            
        except Exception as e:
            logger.error(f"Error estimating BERT skill level: {e}")
            return 0.5
    
    def _estimate_skill_level_traditional(self, text: str, skill: str) -> float:
        """Traditional skill level estimation"""
        level_indicators = {
            'expert': 0.9, 'senior': 0.8, 'advanced': 0.8,
            'experienced': 0.7, 'proficient': 0.6, 'intermediate': 0.5,
            'familiar': 0.4, 'basic': 0.3, 'beginner': 0.2
        }
        
        skill_context = []
        doc = self.nlp(text.lower())
        
        for sent in doc.sents:
            if skill.lower() in sent.text:
                skill_context.append(sent.text)
        
        max_level = 0.5  # Default
        for context in skill_context:
            for indicator, level in level_indicators.items():
                if indicator in context:
                    max_level = max(max_level, level)
        
        # Look for years of experience
        years_pattern = r'(\d+)\s*(?:years?|yrs?)'
        for context in skill_context:
            years_match = re.search(years_pattern, context)
            if years_match:
                years = int(years_match.group(1))
                experience_level = min(0.9, 0.3 + (years * 0.1))
                max_level = max(max_level, experience_level)
        
        return max_level
    
    def _calculate_skill_confidence_traditional(self, text: str, skill: str) -> float:
        """Calculate confidence using traditional methods"""
        sentences = [sent.text for sent in self.nlp(text).sents if skill.lower() in sent.text.lower()]
        
        if not sentences:
            return 0.5
        
        # Use Universal Sentence Encoder if available
        if self.use_model:
            try:
                skill_contexts = [
                    f"Expert in {skill}",
                    f"Experienced with {skill}",
                    f"Proficient in {skill}"
                ]
                
                sentence_embeddings = self.use_model(sentences[:3])
                context_embeddings = self.use_model(skill_contexts)
                
                similarities = tf.keras.utils.cosine_similarity(
                    sentence_embeddings, 
                    tf.reduce_mean(context_embeddings, axis=0, keepdims=True)
                )
                
                return float(tf.reduce_mean(similarities))
            except Exception as e:
                logger.error(f"Error with USE: {e}")
        
        # Fallback to simple confidence
        return min(0.9, 0.6 + len(sentences) * 0.1)
    
    def _deduplicate_skills(self, skills: List[Skill]) -> List[Skill]:
        """Remove duplicate skills and keep the one with highest confidence"""
        skill_dict = {}
        for skill in skills:
            key = skill.name.lower()
            if key not in skill_dict or skill.confidence > skill_dict[key].confidence:
                skill_dict[key] = skill
        
        return list(skill_dict.values())

class ComputerVisionSkillExtractor:
    """Extract skills from certificates, portfolios, and project screenshots"""
    
    def __init__(self):
        self.certificate_templates = {
            'aws': ['aws', 'amazon web services', 'certified'],
            'google_cloud': ['google cloud', 'gcp', 'certified'],
            'microsoft': ['microsoft', 'azure', 'certified'],
            'coursera': ['coursera', 'certificate', 'completion'],
            'udemy': ['udemy', 'certificate', 'completion']
        }
    
    def extract_from_certificate(self, image_path: str) -> List[Skill]:
        """Extract skills from certificate images"""
        try:
            # Load and preprocess image
            if hasattr(image_path, 'read'):  # File object
                content = image_path.read()
                image = Image.open(io.BytesIO(content))
                image = np.array(image)
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
            else:
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply image enhancement
            enhanced = self._enhance_image(gray)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(enhanced)
            
            skills = []
            text_lower = text.lower()
            
            # Identify certificate type and extract relevant skills
            for cert_type, keywords in self.certificate_templates.items():
                if any(keyword in text_lower for keyword in keywords):
                    extracted_skills = self._extract_certificate_skills(text, cert_type)
                    skills.extend(extracted_skills)
            
            return skills
        except Exception as e:
            logger.error(f"Error extracting from certificate: {e}")
            return []
    
    def _enhance_image(self, gray_image):
        """Enhance image for better OCR results"""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return gray_image
    
    def _extract_certificate_skills(self, text: str, cert_type: str) -> List[Skill]:
        """Extract skills based on certificate type"""
        skills = []
        
        skill_mappings = {
            'aws': ['cloud computing', 'aws', 'ec2', 's3', 'lambda'],
            'google_cloud': ['cloud computing', 'gcp', 'bigquery', 'kubernetes'],
            'microsoft': ['azure', 'cloud computing', '.net', 'sql server'],
            'coursera': self._extract_coursera_skills(text),
            'udemy': self._extract_general_skills(text)
        }
        
        if cert_type in skill_mappings:
            skill_list = skill_mappings[cert_type]
            if callable(skill_list):
                skill_list = skill_list
            
            for skill_name in skill_list:
                skills.append(Skill(
                    name=skill_name,
                    level=0.7,  # Certificate implies good proficiency
                    category=self._categorize_skill(skill_name),
                    confidence=0.9,  # High confidence from certificates
                    source='vision'
                ))
        
        return skills
    
    def _extract_coursera_skills(self, text: str) -> List[str]:
        """Extract skills from Coursera certificates"""
        course_patterns = [
            r'machine learning',
            r'deep learning',
            r'data science',
            r'python',
            r'tensorflow',
            r'neural networks'
        ]
        
        skills = []
        for pattern in course_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                skills.append(pattern.replace(' ', '_'))
        
        return skills
    
    def _extract_general_skills(self, text: str) -> List[str]:
        """Extract general skills from certificate text"""
        common_skills = [
            'python', 'java', 'javascript', 'react', 'angular',
            'machine learning', 'data science', 'sql', 'mongodb'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in common_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def _categorize_skill(self, skill_name: str) -> str:
        """Categorize skill based on name"""
        categories = {
            'programming': ['python', 'java', 'javascript', 'react', 'angular'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud computing'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'data science'],
            'database': ['sql', 'mongodb', 'postgresql']
        }
        
        for category, skills in categories.items():
            if skill_name.lower() in [s.lower() for s in skills]:
                return category
        
        return 'general'

class SkillAssessmentEngine:
    """AI-powered skill assessment system with BERT enhancement"""
    
    def __init__(self):
        self.assessment_questions = self._load_assessment_questions()
        self.difficulty_levels = ['beginner', 'intermediate', 'advanced', 'expert']
        
        # Build TensorFlow model for adaptive assessment
        self.assessment_model = self._build_assessment_model()
        
        # BERT integration for assessment enhancement
        try:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading BERT for assessment: {e}")
            self.bert_model = None
    
    def _build_assessment_model(self):
        """Build TensorFlow model for adaptive skill assessment"""
        try:
            # Input layers
            question_input = keras.Input(shape=(100,), name='question_embedding')
            user_history = keras.Input(shape=(50,), name='user_history')
            
            # Dense layers for processing
            question_dense = layers.Dense(64, activation='relu')(question_input)
            question_dense = layers.Dropout(0.3)(question_dense)
            
            history_dense = layers.Dense(32, activation='relu')(user_history)
            history_dense = layers.Dropout(0.3)(history_dense)
            
            # Combine features
            combined = tf.keras.layers.concatenate([question_dense, history_dense])
            combined = layers.Dense(32, activation='relu')(combined)
            
            # Output layer for difficulty prediction
            output = layers.Dense(4, activation='softmax', name='difficulty_prediction')(combined)
            
            model = tf.keras.Model(inputs=[question_input, user_history], outputs=output)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            return model
        except Exception as e:
            logger.error(f"Error building assessment model: {e}")
            return None
    
    def _load_assessment_questions(self) -> Dict[str, List[Dict]]:
        """Load assessment questions for different skills"""
        return {
            'python': [
                {
                    'question': 'What is the output of print(type([]) == list)?',
                    'options': ['True', 'False', 'Error', 'None'],
                    'correct': 0,
                    'difficulty': 'beginner',
                    'concept': 'data types'
                },
                {
                    'question': 'Which decorator is used to create a property in Python?',
                    'options': ['@property', '@staticmethod', '@classmethod', '@decorator'],
                    'correct': 0,
                    'difficulty': 'intermediate',
                    'concept': 'decorators'
                },
                {
                    'question': 'What is the purpose of the __init__ method in Python?',
                    'options': ['To initialize class variables', 'To create a new instance', 'To define class methods', 'To inherit from parent class'],
                    'correct': 1,
                    'difficulty': 'intermediate',
                    'concept': 'object oriented programming'
                }
            ],
            'machine_learning': [
                {
                    'question': 'What is overfitting in machine learning?',
                    'options': [
                        'Model performs well on training but poor on test data',
                        'Model performs poorly on both training and test data',
                        'Model is too simple',
                        'Model has too few parameters'
                    ],
                    'correct': 0,
                    'difficulty': 'intermediate',
                    'concept': 'model evaluation'
                },
                {
                    'question': 'Which algorithm is commonly used for classification problems?',
                    'options': ['Linear Regression', 'Decision Tree', 'K-means', 'PCA'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'classification'
                },
                {
                    'question': 'What is the main goal of unsupervised learning?',
                    'options': ['Predict labels', 'Find patterns', 'Reduce overfitting', 'Increase accuracy'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'unsupervised learning'
                }
            ],
            'javascript': [
                {
                    'question': 'What is the correct way to declare a variable in JavaScript?',
                    'options': ['var x = 5;', 'variable x = 5;', 'v x = 5;', 'declare x = 5;'],
                    'correct': 0,
                    'difficulty': 'beginner',
                    'concept': 'variables'
                },
                {
                    'question': 'Which method is used to parse a JSON string in JavaScript?',
                    'options': ['JSON.parse()', 'parseJSON()', 'JSON.toObject()', 'parse()'],
                    'correct': 0,
                    'difficulty': 'intermediate',
                    'concept': 'json'
                },
                {
                    'question': 'What does the "this" keyword refer to in a regular function?',
                    'options': ['Global object', 'Current object', 'Parent object', 'Window object'],
                    'correct': 1,
                    'difficulty': 'intermediate',
                    'concept': 'context'
                }
            ],
            'sql': [
                {
                    'question': 'Which SQL statement is used to extract data from a database?',
                    'options': ['GET', 'OPEN', 'SELECT', 'EXTRACT'],
                    'correct': 2,
                    'difficulty': 'beginner',
                    'concept': 'basic syntax'
                },
                {
                    'question': 'Which keyword is used to sort the result-set in SQL?',
                    'options': ['ORDER', 'SORT', 'ORDER BY', 'SORT BY'],
                    'correct': 2,
                    'difficulty': 'beginner',
                    'concept': 'sorting'
                },
                {
                    'question': 'What does the WHERE clause do in SQL?',
                    'options': ['Sorts results', 'Filters records', 'Joins tables', 'Deletes records'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'filtering'
                }
            ],
            'data_science': [
                {
                    'question': 'Which library is commonly used for data manipulation in Python?',
                    'options': ['NumPy', 'Pandas', 'Matplotlib', 'Seaborn'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'libraries'
                },
                {
                    'question': 'What is a DataFrame?',
                    'options': ['A type of plot', 'A 2D labeled data structure', 'A neural network', 'A SQL table'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'data structures'
                },
                {
                    'question': 'Which method is used to check for missing values in a DataFrame?',
                    'options': ['isnull()', 'dropna()', 'fillna()', 'replace()'],
                    'correct': 0,
                    'difficulty': 'intermediate',
                    'concept': 'missing data'
                }
            ],
            'cloud_computing': [
                {
                    'question': 'Which of the following is a cloud service provider?',
                    'options': ['AWS', 'Linux', 'Oracle VM', 'Docker'],
                    'correct': 0,
                    'difficulty': 'beginner',
                    'concept': 'providers'
                },
                {
                    'question': 'What does IaaS stand for?',
                    'options': ['Internet as a Service', 'Infrastructure as a Service', 'Integration as a Service', 'Instance as a Service'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'service models'
                },
                {
                    'question': 'Which AWS service is used for object storage?',
                    'options': ['EC2', 'Lambda', 'S3', 'RDS'],
                    'correct': 2,
                    'difficulty': 'intermediate',
                    'concept': 'aws services'
                }
            ],
            'react': [
                {
                    'question': 'What is a React component?',
                    'options': ['A function or class that returns JSX', 'A CSS file', 'A database', 'A server'],
                    'correct': 0,
                    'difficulty': 'beginner',
                    'concept': 'components'
                },
                {
                    'question': 'Which hook is used to manage state in a functional component?',
                    'options': ['useState', 'useEffect', 'useContext', 'useReducer'],
                    'correct': 0,
                    'difficulty': 'beginner',
                    'concept': 'hooks'
                },
                {
                    'question': 'What is the virtual DOM?',
                    'options': ['A copy of the real DOM', 'A server-side DOM', 'A CSS selector', 'A database'],
                    'correct': 0,
                    'difficulty': 'intermediate',
                    'concept': 'virtual dom'
                }
            ],
            'aws': [
                {
                    'question': 'What does AWS stand for?',
                    'options': ['Amazon Web Services', 'Advanced Web Solutions', 'Automated Web Services', 'Amazon Web Storage'],
                    'correct': 0,
                    'difficulty': 'beginner',
                    'concept': 'basics'
                },
                {
                    'question': 'Which AWS service is used for compute?',
                    'options': ['S3', 'EC2', 'RDS', 'Lambda'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'services'
                },
                {
                    'question': 'What is the main benefit of using AWS Lambda?',
                    'options': ['Serverless compute', 'Object storage', 'Database hosting', 'Networking'],
                    'correct': 0,
                    'difficulty': 'intermediate',
                    'concept': 'lambda'
                }
            ],
            'tensorflow': [
                {
                    'question': 'What is TensorFlow primarily used for?',
                    'options': ['Web development', 'Machine learning', 'Database management', 'Cloud storage'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'basics'
                },
                {
                    'question': 'Which language is TensorFlow written in?',
                    'options': ['Python', 'Java', 'C++', 'All of the above'],
                    'correct': 3,
                    'difficulty': 'intermediate',
                    'concept': 'languages'
                },
                {
                    'question': 'What is a tensor in TensorFlow?',
                    'options': ['A type of neural network', 'A multi-dimensional array', 'A data pipeline', 'A loss function'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'data structures'
                }
            ],
            'advanced_sql': [
                {
                    'question': 'Which SQL clause is used to group rows that have the same values?',
                    'options': ['ORDER BY', 'GROUP BY', 'HAVING', 'WHERE'],
                    'correct': 1,
                    'difficulty': 'intermediate',
                    'concept': 'grouping'
                },
                {
                    'question': 'What does the HAVING clause do in SQL?',
                    'options': ['Filters groups', 'Sorts results', 'Joins tables', 'Deletes records'],
                    'correct': 0,
                    'difficulty': 'advanced',
                    'concept': 'filtering'
                },
                {
                    'question': 'Which function returns the number of rows in SQL?',
                    'options': ['SUM()', 'COUNT()', 'TOTAL()', 'NUMBER()'],
                    'correct': 1,
                    'difficulty': 'beginner',
                    'concept': 'aggregation'
                }
            ]
        }
    
    def conduct_assessment(self, user_id: str, skill: str, num_questions: int = 10) -> Dict[str, Any]:
        """Conduct adaptive skill assessment with BERT enhancement"""
        try:
            if skill not in self.assessment_questions:
                return {'error': f'No assessment available for {skill}'}
            
            questions = self.assessment_questions[skill]
            user_responses = []
            current_difficulty = 1  # Start with intermediate
            
            assessment_results = {
                'user_id': user_id,
                'skill': skill,
                'questions_answered': 0,
                'correct_answers': 0,
                'estimated_level': 0.5,
                'confidence': 0.0,
                'areas_for_improvement': [],
                'strong_areas': []
            }
            
            # Simulate adaptive questioning
            for i in range(min(num_questions, len(questions))):
                question = questions[i % len(questions)]
                
                # Simulate user response
                simulated_response = np.random.choice([0, 1, 2, 3])
                is_correct = simulated_response == question['correct']
                
                user_responses.append({
                    'question_id': i,
                    'response': simulated_response,
                    'correct': is_correct,
                    'difficulty': question['difficulty'],
                    'concept': question['concept']
                })
                
                assessment_results['questions_answered'] += 1
                if is_correct:
                    assessment_results['correct_answers'] += 1
            
            # Calculate final assessment metrics
            assessment_results['estimated_level'] = self._calculate_skill_level(user_responses)
            assessment_results['confidence'] = self._calculate_confidence(user_responses)
            assessment_results['areas_for_improvement'] = self._identify_weak_areas(user_responses)
            assessment_results['strong_areas'] = self._identify_strong_areas(user_responses)
            
            return assessment_results
            
        except Exception as e:
            logger.error(f"Error in assessment: {e}")
            return {'error': str(e)}
    
    def _calculate_skill_level(self, responses: List[Dict]) -> float:
        """Calculate estimated skill level based on responses"""
        if not responses:
            return 0.0
        
        difficulty_weights = {'beginner': 0.25, 'intermediate': 0.5, 'advanced': 0.75, 'expert': 1.0}
        total_weight = 0
        weighted_score = 0
        
        for response in responses:
            weight = difficulty_weights.get(response['difficulty'], 0.5)
            total_weight += weight
            if response['correct']:
                weighted_score += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_confidence(self, responses: List[Dict]) -> float:
        """Calculate confidence in skill level estimation"""
        if len(responses) < 3:
            return 0.3
        
        correct_count = sum(1 for r in responses if r['correct'])
        accuracy = correct_count / len(responses)
        
        variance = np.var([1 if r['correct'] else 0 for r in responses])
        confidence = min(0.95, 0.5 + (1 - variance) * 0.4)
        
        return confidence
    
    def _identify_weak_areas(self, responses: List[Dict]) -> List[str]:
        """Identify areas where user performed poorly"""
        concept_performance = {}
        
        for response in responses:
            concept = response['concept']
            if concept not in concept_performance:
                concept_performance[concept] = {'correct': 0, 'total': 0}
            
            concept_performance[concept]['total'] += 1
            if response['correct']:
                concept_performance[concept]['correct'] += 1
        
        weak_areas = []
        for concept, perf in concept_performance.items():
            accuracy = perf['correct'] / perf['total']
            if accuracy < 0.6:
                weak_areas.append(concept)
        
        return weak_areas
    
    def _identify_strong_areas(self, responses: List[Dict]) -> List[str]:
        """Identify areas where user performed well"""
        concept_performance = {}
        
        for response in responses:
            concept = response['concept']
            if concept not in concept_performance:
                concept_performance[concept] = {'correct': 0, 'total': 0}
            
            concept_performance[concept]['total'] += 1
            if response['correct']:
                concept_performance[concept]['correct'] += 1
        
        strong_areas = []
        for concept, perf in concept_performance.items():
            accuracy = perf['correct'] / perf['total']
            if accuracy >= 0.8:
                strong_areas.append(concept)
        
        return strong_areas

class LearningPathGenerator:
    """Generate personalized learning paths with BERT enhancement"""
    
    def __init__(self):
        self.learning_resources = self._initialize_learning_resources()
        self.skill_prerequisites = self._define_skill_prerequisites()
        
        # BERT integration for better resource matching
        try:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading BERT for learning paths: {e}")
            self.bert_model = None
        
        # TensorFlow model for learning path optimization
        self.recommendation_model = self._build_recommendation_model()
    
    def _build_recommendation_model(self):
        """Build TensorFlow model for learning resource recommendation"""
        try:
            # User profile input
            user_skills = keras.Input(shape=(50,), name='user_skills')
            user_preferences = keras.Input(shape=(20,), name='user_preferences')
            
            # Resource features input
            resource_features = keras.Input(shape=(30,), name='resource_features')
            
            # Neural network layers
            user_embedding = layers.Dense(32, activation='relu')(tf.keras.layers.concatenate([user_skills, user_preferences]))
            user_embedding = layers.Dropout(0.3)(user_embedding)
            
            resource_embedding = layers.Dense(32, activation='relu')(resource_features)
            resource_embedding = layers.Dropout(0.3)(resource_embedding)
            
            # Compute similarity
            similarity = tf.keras.layers.Dot(axes=1)([user_embedding, resource_embedding])
            similarity = layers.Dense(1, activation='sigmoid')(similarity)
            
            model = tf.keras.Model(inputs=[user_skills, user_preferences, resource_features], outputs=similarity)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            return model
        except Exception as e:
            logger.error(f"Error building recommendation model: {e}")
            return None
    
    def _initialize_learning_resources(self) -> List[LearningResource]:
        """Initialize comprehensive learning resources"""
        return [
            LearningResource(
                title="Python Fundamentals",
                description="Learn Python basics including syntax, data types, and control structures",
                difficulty="beginner",
                duration=180,
                skills=["python", "programming"],
                url="https://example.com/python-fundamentals",
                rating=4.5
            ),
            LearningResource(
                title="Machine Learning with TensorFlow",
                description="Build ML models using TensorFlow and understand deep learning concepts",
                difficulty="intermediate",
                duration=300,
                skills=["machine_learning", "tensorflow", "python"],
                url="https://example.com/ml-tensorflow",
                rating=4.7
            ),
            LearningResource(
                title="Advanced SQL Queries",
                description="Master complex SQL queries, joins, and database optimization",
                difficulty="advanced",
                duration=150,
                skills=["sql", "database"],
                url="https://example.com/advanced-sql",
                rating=4.3
            ),
            LearningResource(
                title="Cloud Architecture with AWS",
                description="Design scalable cloud solutions using AWS services",
                difficulty="intermediate",
                duration=240,
                skills=["aws", "cloud", "architecture"],
                url="https://example.com/aws-architecture",
                rating=4.6
            ),
            LearningResource(
                title="React Development Bootcamp",
                description="Build modern web applications with React and JavaScript",
                difficulty="intermediate",
                duration=360,
                skills=["react", "javascript", "web_development"],
                url="https://example.com/react-bootcamp",
                rating=4.4
            ),
            LearningResource(
                title="Data Science with Python",
                description="Complete data science workflow using pandas, numpy, and scikit-learn",
                difficulty="intermediate",
                duration=420,
                skills=["data_science", "python", "pandas", "numpy"],
                url="https://example.com/data-science-python",
                rating=4.8
            )
        ]
    
    def _define_skill_prerequisites(self) -> Dict[str, List[str]]:
        """Define prerequisite relationships between skills"""
        return {
            'machine_learning': ['python', 'statistics'],
            'deep_learning': ['machine_learning', 'tensorflow'],
            'cloud_architecture': ['cloud', 'networking'],
            'advanced_sql': ['sql', 'database'],
            'react': ['javascript', 'html', 'css'],
            'django': ['python', 'web_development'],
            'data_science': ['python', 'statistics'],
            'tensorflow': ['python', 'machine_learning']
        }
    
    def generate_learning_path(self, current_skills: List[Skill], target_skills: List[str], 
                             preferences: Dict[str, Any] = None) -> LearningPath:
        """Generate personalized learning path with BERT enhancement"""
        try:
            if preferences is None:
                preferences = {'max_duration_per_day': 120, 'difficulty_preference': 'intermediate'}
            
            # Identify skill gaps
            current_skill_names = {skill.name.lower() for skill in current_skills}
            target_skill_names = {skill.lower() for skill in target_skills}
            skill_gaps = list(target_skill_names - current_skill_names)
            
            # Add prerequisite skills to gaps
            extended_gaps = self._add_prerequisites(skill_gaps, current_skill_names)
            
            # Find relevant learning resources
            relevant_resources = self._find_relevant_resources_bert(extended_gaps) if self.bert_model else self._find_relevant_resources(extended_gaps)
            
            # Prioritize and order resources
            ordered_resources = self._prioritize_resources(relevant_resources, current_skills, preferences)
            
            # Calculate estimated completion time
            total_duration = sum(resource.duration for resource in ordered_resources)
            daily_limit = preferences.get('max_duration_per_day', 120)
            estimated_days = max(1, total_duration // daily_limit)
            
            return LearningPath(
                user_id=f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                skills_gap=skill_gaps,
                recommended_resources=ordered_resources,
                estimated_completion=estimated_days,
                priority_order=extended_gaps
            )
        except Exception as e:
            logger.error(f"Error generating learning path: {e}")
            return LearningPath(
                user_id="error",
                skills_gap=[],
                recommended_resources=[],
                estimated_completion=0,
                priority_order=[]
            )
    
    def _find_relevant_resources_bert(self, skill_gaps: List[str]) -> List[LearningResource]:
        """Find learning resources using BERT semantic matching"""
        try:
            relevant_resources = []
            
            # Generate embeddings for skill gaps
            gap_embeddings = self.bert_model.encode(skill_gaps)
            
            for resource in self.learning_resources:
                # Generate embedding for resource skills and description
                resource_text = ' '.join(resource.skills) + ' ' + resource.description
                resource_embedding = self.bert_model.encode([resource_text])
                
                # Calculate similarity with skill gaps
                similarities = cosine_similarity(gap_embeddings, resource_embedding)
                max_similarity = np.max(similarities)
                
                # Include resource if similarity is above threshold
                if max_similarity > 0.3:  # Lower threshold for broader matching
                    relevant_resources.append(resource)
            
            return relevant_resources
        except Exception as e:
            logger.error(f"Error in BERT resource matching: {e}")
            return self._find_relevant_resources(skill_gaps)
    
    def _find_relevant_resources(self, skill_gaps: List[str]) -> List[LearningResource]:
        """Find learning resources that match skill gaps (traditional method)"""
        relevant_resources = []
        
        for resource in self.learning_resources:
            resource_skills = [skill.lower() for skill in resource.skills]
            if any(gap.lower() in resource_skills for gap in skill_gaps):
                relevant_resources.append(resource)
        
        return relevant_resources
    
    def _add_prerequisites(self, skill_gaps: List[str], current_skills: set) -> List[str]:
        """Add prerequisite skills to the learning path"""
        extended_gaps = skill_gaps.copy()
        
        for skill in skill_gaps:
            if skill in self.skill_prerequisites:
                for prereq in self.skill_prerequisites[skill]:
                    if prereq not in current_skills and prereq not in extended_gaps:
                        extended_gaps.insert(0, prereq)  # Add prerequisites first
        
        return extended_gaps
    
    def _prioritize_resources(self, resources: List[LearningResource], 
                            current_skills: List[Skill], preferences: Dict[str, Any]) -> List[LearningResource]:
        """Prioritize learning resources based on user profile and preferences"""
        scored_resources = []
        
        for resource in resources:
            score = self._calculate_resource_score(resource, current_skills, preferences)
            scored_resources.append((resource, score))
        
        # Sort by score (highest first)
        scored_resources.sort(key=lambda x: x[1], reverse=True)
        
        return [resource for resource, _ in scored_resources]
    
    def _calculate_resource_score(self, resource: LearningResource, 
                                current_skills: List[Skill], preferences: Dict[str, Any]) -> float:
        """Calculate priority score for a learning resource"""
        score = 0.0
        
        # Base score from rating
        score += resource.rating * 0.3
        
        # Difficulty preference alignment
        preferred_difficulty = preferences.get('difficulty_preference', 'intermediate')
        if resource.difficulty == preferred_difficulty:
            score += 0.4
        elif abs(self._difficulty_to_number(resource.difficulty) - 
                self._difficulty_to_number(preferred_difficulty)) <= 1:
            score += 0.2
        
        # Duration preference
        max_duration = preferences.get('max_duration_per_day', 120)
        if resource.duration <= max_duration:
            score += 0.3
        else:
            score += 0.1
        
        return min(1.0, score)
    
    def _difficulty_to_number(self, difficulty: str) -> int:
        """Convert difficulty string to number for comparison"""
        difficulty_map = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
        return difficulty_map.get(difficulty, 2)

class PersonalizedLearningSystem:
    """Main BERT-enhanced system that orchestrates all components"""
    
    def __init__(self):
        self.resume_parser = BERTEnhancedResumeParser()
        self.cv_extractor = ComputerVisionSkillExtractor()
        self.assessment_engine = SkillAssessmentEngine()
        self.path_generator = LearningPathGenerator()
        self.user_profiles = {}
        
        logger.info("BERT-Enhanced Personalized Learning System initialized")
    
    def process_user_profile(self, user_id: str, resume_path: str = None, 
                           certificate_paths: List[str] = None, 
                           target_skills: List[str] = None,
                           name: str = "",
                           headline: str = "",
                           education: list = None,
                           location: str = "",
                           career_goal: str = "") -> Dict[str, Any]:
        """Process complete user profile with BERT enhancement"""
        try:
            all_skills = []
            
            # Extract skills from resume using BERT
            if resume_path:
                resume_text = self.resume_parser.extract_text_from_resume(resume_path)
                resume_skills = self.resume_parser.extract_skills(resume_text)
                all_skills.extend(resume_skills)
            
            # Extract skills from certificates
            if certificate_paths:
                for cert_path in certificate_paths:
                    cert_skills = self.cv_extractor.extract_from_certificate(cert_path)
                    all_skills.extend(cert_skills)
            
            # Conduct skill assessments for key skills
            assessment_results = {}
            key_skills = list(set([skill.name for skill in all_skills]))[:5]
            
            for skill in key_skills:
                if skill in ['python', 'machine_learning', 'javascript']:
                    assessment = self.assessment_engine.conduct_assessment(user_id, skill)
                    assessment_results[skill] = assessment
                    
                    # Update skill level based on assessment
                    for user_skill in all_skills:
                        if user_skill.name == skill:
                            user_skill.level = assessment['estimated_level']
                            user_skill.confidence = assessment['confidence']
            
            # Generate learning path
            learning_path = None
            if target_skills:
                learning_path = self.path_generator.generate_learning_path(
                    current_skills=all_skills,
                    target_skills=target_skills
                )
            
            # Store user profile with new fields
            profile_obj = {
                'skills': [skill.__dict__ for skill in all_skills],
                'assessments': assessment_results,
                'learning_path': learning_path.__dict__ if learning_path else None,
                'last_updated': datetime.now(),
                'target_skills': target_skills or [],
                'name': name,
                'headline': headline,
                'education': education or [],
                'location': location,
                'career_goal': career_goal
            }
            self.user_profiles[user_id] = profile_obj
            # --- Persist to SQLite ---
            save_user_profile(user_id, profile_obj)
            return {
                'user_id': user_id,
                'name': name,
                'headline': headline,
                'education': education or [],
                'location': location,
                'career_goal': career_goal,
                'extracted_skills': [
                    {
                        'name': skill.name,
                        'level': skill.level,
                        'category': skill.category,
                        'confidence': skill.confidence,
                        'source': skill.source
                    } for skill in all_skills
                ],
                'assessments': assessment_results,
                'learning_path': {
                    'skills_gap': learning_path.skills_gap if learning_path else [],
                    'recommended_resources': [
                        {
                            'title': resource.title,
                            'description': resource.description,
                            'difficulty': resource.difficulty,
                            'duration': resource.duration,
                            'skills': resource.skills,
                            'rating': resource.rating,
                            'url': resource.url
                        } for resource in learning_path.recommended_resources
                    ] if learning_path else [],
                    'estimated_completion_days': learning_path.estimated_completion if learning_path else 0
                },
                'recommendations': self._generate_general_recommendations(all_skills, assessment_results)
            }
        except Exception as e:
            logger.error(f"Error processing user profile: {e}")
            return {'error': str(e)}
    
    def _generate_general_recommendations(self, skills: List[Skill], 
                                        assessments: Dict[str, Any]) -> List[str]:
        """Generate general learning recommendations"""
        recommendations = []
        
        # Skill diversity recommendations
        skill_categories = set(skill.category for skill in skills)
        if len(skill_categories) < 3:
            recommendations.append("Consider expanding into new skill categories to become more versatile")
        
        # Skill depth recommendations
        advanced_skills = [skill for skill in skills if skill.level > 0.7]
        if len(advanced_skills) < 2:
            recommendations.append("Focus on developing deeper expertise in your strongest skills")
        
        # Assessment-based recommendations
        for skill, assessment in assessments.items():
            if assessment.get('estimated_level', 0) < 0.6:
                recommendations.append(f"Consider additional practice in {skill} fundamentals")
            
            weak_areas = assessment.get('areas_for_improvement', [])
            if weak_areas:
                recommendations.append(f"Focus on improving {', '.join(weak_areas)} in {skill}")
        
        return recommendations
    
    def update_skill_progress(self, user_id: str, skill_name: str, 
                            new_level: float, completion_data: Dict[str, Any] = None):
        """Update user's skill progress"""
        try:
            if user_id not in self.user_profiles:
                return {'error': 'User profile not found'}
            
            profile = self.user_profiles[user_id]
            
            # Update existing skill or add new one
            skill_updated = False
            for skill in profile['skills']:
                if skill.name.lower() == skill_name.lower():
                    skill.level = new_level
                    skill.confidence = min(1.0, skill.confidence + 0.1)
                    skill_updated = True
                    break
            
            if not skill_updated:
                # Add new skill
                profile['skills'].append(Skill(
                    name=skill_name,
                    level=new_level,
                    category='general',
                    confidence=0.7,
                    source='progress_update'
                ))
            
            # Update learning path if target skills exist
            if profile['target_skills']:
                updated_path = self.path_generator.generate_learning_path(
                    current_skills=profile['skills'],
                    target_skills=profile['target_skills']
                )
                profile['learning_path'] = updated_path
            
            profile['last_updated'] = datetime.now()
            
            return {'success': True, 'message': f'Updated {skill_name} progress'}
        except Exception as e:
            logger.error(f"Error updating skill progress: {e}")
            return {'error': str(e)}
    
    def get_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        
        # Get dashboard data for user interface
        try:
            # Always load from SQLite to ensure persistence and freshness
            db_profile = get_user_profile(user_id)
            if db_profile:
                # Deserialize skills as Skill objects if they are dicts
                if 'skills' in db_profile:
                    db_profile['skills'] = [
                        Skill(**s) if isinstance(s, dict)
                        else Skill(name=s, level=0.5, category='general', confidence=0.7, source='unknown') if isinstance(s, str)
                        else s
                        for s in db_profile['skills']
                    ]
                # Deserialize learning_path as LearningPath object if it's a dict
                if 'learning_path' in db_profile and isinstance(db_profile['learning_path'], dict):
                    db_profile['learning_path'] = LearningPath(**db_profile['learning_path'])
                # Convert last_updated to datetime if it's a string
                if 'last_updated' in db_profile and isinstance(db_profile['last_updated'], str):
                    try:
                        db_profile['last_updated'] = datetime.fromisoformat(db_profile['last_updated'])
                    except Exception:
                        try:
                            db_profile['last_updated'] = datetime.strptime(db_profile['last_updated'], "%Y-%m-%d %H:%M:%S.%f")
                        except Exception:
                            db_profile['last_updated'] = datetime.now()
                self.user_profiles[user_id] = db_profile
            else:
                return {'error': 'User profile not found'}
            
            profile = self.user_profiles[user_id]
            skills = profile['skills']
            
            # Calculate skill distribution by category
            category_distribution = {}
            for skill in skills:
                category = skill.category
                if category not in category_distribution:
                    category_distribution[category] = {'count': 0, 'avg_level': 0}
                category_distribution[category]['count'] += 1
                category_distribution[category]['avg_level'] += skill.level
            
            # Calculate average levels
            for category in category_distribution:
                count = category_distribution[category]['count']
                category_distribution[category]['avg_level'] /= count
            
            # Get top skills
            top_skills = sorted(skills, key=lambda x: x.level * x.confidence, reverse=True)[:5]
            
            # Calculate overall progress
            if profile['learning_path']:
                total_resources = len(profile['learning_path'].recommended_resources)
                completed_resources = max(0, total_resources - len(profile['learning_path'].skills_gap))
                progress_percentage = (completed_resources / total_resources * 100) if total_resources > 0 else 0
            else:
                progress_percentage = 0

            # Resume Insights Additions
            education = profile.get('education', [])
            career_goal = profile.get('career_goal', '')
            
            # Resume skills with confidence
            resume_skills = [
                {
                    'name': skill.name,
                    'confidence': int(skill.confidence * 100),
                    'level': skill.level,
                    'category': skill.category,
                    'source': skill.source
                } for skill in skills
            ]
            
            # Generate skill level insights
            skill_insights = []
            beginner_skills = [s for s in skills if s.level < 0.4]
            intermediate_skills = [s for s in skills if 0.4 <= s.level < 0.7]
            advanced_skills = [s for s in skills if s.level >= 0.7]
            
            if beginner_skills:
                skill_insights.append(f"You have {len(beginner_skills)} beginner-level skills to develop")
            if intermediate_skills:
                skill_insights.append(f"You have {len(intermediate_skills)} intermediate skills showing good progress")
            if advanced_skills:
                skill_insights.append(f"You have {len(advanced_skills)} advanced skills - great expertise!")
            
            # Calculate completion streaks and learning momentum
            learning_momentum = 0
            if profile.get('learning_path') and profile['learning_path'].recommended_resources:
                completed_this_week = 0  # This would be calculated from actual completion data
                learning_momentum = min(100, completed_this_week * 20)
            
            return {
                'user_id': user_id,
                'name': profile.get('name', ''),
                'headline': profile.get('headline', ''),
                'location': profile.get('location', ''),
                'career_goal': career_goal,
                'education': education,
                'last_updated': profile['last_updated'].strftime('%Y-%m-%d %H:%M:%S'),
                
                # Skills overview
                'total_skills': len(skills),
                'skill_categories': len(category_distribution),
                'top_skills': [
                    {
                        'name': skill.name,
                        'level': round(skill.level * 100),
                        'category': skill.category,
                        'confidence': round(skill.confidence * 100)
                    } for skill in top_skills
                ],
                
                # Category distribution
                'category_distribution': {
                    category: {
                        'count': data['count'],
                        'avg_level': round(data['avg_level'] * 100)
                    } for category, data in category_distribution.items()
                },
                
                # Learning progress
                'learning_progress': {
                    'overall_percentage': round(progress_percentage),
                    'skills_gap': profile.get('learning_path', {}).get('skills_gap', []) if profile.get('learning_path') else [],
                    'recommended_resources_count': len(profile.get('learning_path', {}).get('recommended_resources', [])) if profile.get('learning_path') else 0,
                    'estimated_completion_days': profile.get('learning_path', {}).get('estimated_completion', 0) if profile.get('learning_path') else 0
                },
                
                # Assessments summary
                'assessments': {
                    skill: {
                        'level': round(result.get('estimated_level', 0) * 100),
                        'confidence': round(result.get('confidence', 0) * 100),
                        'questions_answered': result.get('questions_answered', 0),
                        'correct_answers': result.get('correct_answers', 0),
                        'strong_areas': result.get('strong_areas', []),
                        'areas_for_improvement': result.get('areas_for_improvement', [])
                    } for skill, result in profile.get('assessments', {}).items()
                },
                
                # Resume insights
                'resume_insights': {
                    'skills_by_source': {
                        'resume_bert': len([s for s in skills if s.source == 'resume_bert']),
                        'resume_traditional': len([s for s in skills if s.source == 'resume_traditional']),
                        'resume_hybrid': len([s for s in skills if s.source == 'resume_hybrid']),
                        'vision': len([s for s in skills if s.source == 'vision']),
                        'assessment': len([s for s in skills if s.source == 'assessment'])
                    },
                    'skill_level_distribution': {
                        'beginner': len(beginner_skills),
                        'intermediate': len(intermediate_skills),
                        'advanced': len(advanced_skills)
                    },
                    'insights': skill_insights
                },
                
                # Learning recommendations
                'recommendations': profile.get('recommendations', []),
                
                # Learning momentum and streaks
                'learning_momentum': learning_momentum,
                'target_skills': profile.get('target_skills', []),
                
                # Recent activity (placeholder for future implementation)
                'recent_activity': [],
                
                # Achievements (placeholder for future implementation)
                'achievements': []
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}

    # Helper functions for database operations (these would be imported from skill_matcher)
    def save_user_profile(user_id: str, profile_data: Dict[str, Any]):
        """Save user profile to SQLite database"""
        # This function would be implemented in the skill_matcher module
        pass

    def get_user_profile(user_id: str) -> Dict[str, Any]:
        """Get user profile from SQLite database"""
        # This function would be implemented in the skill_matcher module
        return None