import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
# from sentence_transformers import SentenceTransformer  # moved to lazy property
# import torch  # moved to lazy function if needed
from typing import List, Dict, Optional, Tuple, Any
import pickle
import logging
from functools import cached_property
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LazyBERTJobRecommenderSystem:
    """
    Enhanced Job Recommender System with Lazy Loading and BERT integration
    
    Key Features:
    - Lazy initialization of BERT model and embeddings
    - Memory-efficient loading patterns
    - Thread-safe lazy loading
    - Cached properties for expensive operations
    """
    
    def __init__(self, jobs_df, user_interactions_df=None, bert_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the recommender system with lazy loading
        
        Parameters:
        jobs_df (DataFrame): DataFrame containing job listings
        user_interactions_df (DataFrame): Optional DataFrame containing user-job interactions
        bert_model_name (str): Name of the BERT model to use for embeddings
        """
        self.jobs_df = jobs_df.copy()
        
        # Adding a job ID if not present
        if 'Job_ID' not in self.jobs_df.columns:
            self.jobs_df['Job_ID'] = range(len(self.jobs_df))
        
        self.user_interactions_df = user_interactions_df
        self.bert_model_name = bert_model_name
        
        # Lazy-loaded components (initialized as None)
        self._bert_model = None
        self._job_embeddings = None
        self._user_item_matrix = None
        self._similarity_matrix = None
        self._preprocessed_data = None
        
        # Configuration
        self.hybrid_weights = {'content': 0.6, 'collaborative': 0.4}
        self.embedding_batch_size = 32
        
        # Thread safety for lazy loading
        self._bert_lock = threading.Lock()
        self._embeddings_lock = threading.Lock()
        self._collaborative_lock = threading.Lock()
        self._preprocessing_lock = threading.Lock()
        
        # Cache for frequently accessed data
        self._cache = weakref.WeakValueDictionary()
        
        logger.info("LazyBERTJobRecommenderSystem initialized (models not loaded yet)")
    
    @cached_property
    def bert_model(self) -> "SentenceTransformer":
        """Lazy-loaded BERT model with thread safety"""
        if self._bert_model is None:
            with self._bert_lock:
                if self._bert_model is None:  # Double-check pattern
                    logger.info(f"Lazy loading BERT model: {self.bert_model_name}")
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._bert_model = SentenceTransformer(self.bert_model_name)
                        logger.info("BERT model loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading BERT model: {e}")
                        raise
        return self._bert_model
    
    def _ensure_torch(self):
        import torch
        return torch
    
    @cached_property
    def preprocessed_data(self) -> pd.DataFrame:
        """Lazy-loaded preprocessed data"""
        if self._preprocessed_data is None:
            with self._preprocessing_lock:
                if self._preprocessed_data is None:
                    logger.info("Lazy loading and preprocessing data...")
                    self._preprocessed_data = self._preprocess_data()
        return self._preprocessed_data
    
    def _preprocess_data(self) -> pd.DataFrame:
        """Internal method to preprocess job data"""
        df = self.jobs_df.copy()
        
        # Handle salary - convert to numeric, or create a binary indicator
        if 'Job Salary' in df.columns:
            try:
                df['Salary_Numeric'] = pd.to_numeric(df['Job Salary'], errors='coerce')
                
                if df['Salary_Numeric'].isna().all():
                    logger.warning("Could not convert salary values to numeric. Using binary indicator.")
                    df['Salary_Disclosed'] = df['Job Salary'].apply(
                        lambda x: 0 if str(x).lower().strip() in ['not disclosed', 'not disclosed by recruiter', 'na', 'n/a', ''] else 1
                    )
                else:
                    median_salary = df['Salary_Numeric'].median()
                    df['Salary_Numeric'] = df['Salary_Numeric'].fillna(median_salary)
                    
                    # Normalize the numeric salary
                    scaler = MinMaxScaler()
                    df['Normalized_Salary'] = scaler.fit_transform(df[['Salary_Numeric']])
            except Exception as e:
                logger.warning(f"Error processing salary data: {e}")
        
        # Handle experience - convert to numeric if possible
        if 'Job Experience Required' in df.columns:
            try:
                def extract_years(exp_text):
                    if pd.isna(exp_text) or not isinstance(exp_text, str):
                        return 0
                    
                    exp_text = exp_text.lower().strip()
                    if '-' in exp_text:
                        parts = exp_text.split('-')
                        if len(parts) == 2:
                            try:
                                min_exp = float(''.join(c for c in parts[0] if c.isdigit() or c == '.'))
                                max_exp = float(''.join(c for c in parts[1] if c.isdigit() or c == '.'))
                                return (min_exp + max_exp) / 2
                            except ValueError:
                                return 0
                    else:
                        try:
                            return float(''.join(c for c in exp_text if c.isdigit() or c == '.'))
                        except ValueError:
                            return 0
                
                df['Experience_Numeric'] = df['Job Experience Required'].apply(extract_years)
                
                # Normalize the numeric experience
                scaler = MinMaxScaler()
                df['Normalized_Experience'] = scaler.fit_transform(df[['Experience_Numeric']])
            except Exception as e:
                logger.warning(f"Error processing experience data: {e}")
        
        # Create combined text feature for BERT embedding
        text_features = []
        for col in ['Job Title', 'Key Skills', 'Role Category', 'Functional Area', 'Industry']:
            if col in df.columns:
                text_features.append(df[col].fillna('').astype(str))
        
        # Combine all text features
        if text_features:
            df['Combined_Features'] = pd.Series(
                ' '.join(str(val) for val in vals) for vals in zip(*text_features)
            )
        else:
            logger.warning("No text features found for content-based filtering")
            df['Combined_Features'] = ""
        
        return df
    
    @cached_property
    def job_embeddings(self) -> np.ndarray:
        """Lazy-loaded job embeddings with batch processing"""
        if self._job_embeddings is None:
            with self._embeddings_lock:
                if self._job_embeddings is None:
                    logger.info("Lazy loading BERT embeddings...")
                    self._job_embeddings = self._build_embeddings()
        return self._job_embeddings
    
    def _build_embeddings(self) -> np.ndarray:
        """Internal method to build BERT embeddings"""
        # Ensure preprocessed data is available
        df = self.preprocessed_data
        
        # Get job descriptions
        job_descriptions = df['Combined_Features'].tolist()
        
        # Generate embeddings in batches to handle memory efficiently
        embeddings = []
        batch_size = self.embedding_batch_size
        
        for i in range(0, len(job_descriptions), batch_size):
            batch = job_descriptions[i:i + batch_size]
            batch_embeddings = self.bert_model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(job_descriptions)} jobs")
        
        # Concatenate all embeddings
        result = np.vstack(embeddings)
        logger.info(f"BERT embeddings created with shape: {result.shape}")
        return result
    
    @cached_property
    def collaborative_model(self) -> Optional[pd.DataFrame]:
        """Lazy-loaded collaborative filtering model"""
        if self.user_interactions_df is None:
            logger.warning("No user interaction data available for collaborative filtering")
            return None
        
        if self._similarity_matrix is None:
            with self._collaborative_lock:
                if self._similarity_matrix is None:
                    logger.info("Lazy loading collaborative filtering model...")
                    self._build_collaborative_model()
        
        return self._similarity_matrix
    
    def _build_collaborative_model(self):
        """Internal method to build collaborative filtering model"""
        if self.user_interactions_df is None:
            return
        
        # Create user-item matrix
        user_item_matrix = pd.pivot_table(
            self.user_interactions_df,
            values='Rating',
            index='User_ID',
            columns='Job_ID',
            fill_value=0
        )
        
        self._user_item_matrix = user_item_matrix
        
        # Calculate item-item similarity matrix
        item_item_similarity = cosine_similarity(user_item_matrix.T)
        self._similarity_matrix = pd.DataFrame(
            item_item_similarity,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )
        
        logger.info("Collaborative filtering model built successfully")
    
    def get_bert_recommendations(self, job_id=None, user_profile_text=None, top_n=5) -> pd.DataFrame:
        """
        Get recommendations using lazy-loaded BERT embeddings
        
        Parameters:
        job_id: ID of a job to find similar jobs for
        user_profile_text: Text description of user profile/requirements
        top_n: Number of recommendations to return
        """
        # Lazy load embeddings only when needed
        embeddings = self.job_embeddings
        df = self.preprocessed_data
        
        if job_id is not None:
            # Job-to-job similarity
            if job_id not in df['Job_ID'].values:
                logger.error(f"Job ID {job_id} not found in dataset")
                return pd.DataFrame()
            
            # Find the index of the job
            job_idx = df[df['Job_ID'] == job_id].index[0]
            
            # Calculate similarity with all other jobs
            similarities = cosine_similarity(
                embeddings[job_idx:job_idx+1], 
                embeddings
            ).flatten()
            
            # Get top similar jobs (excluding the input job)
            similar_indices = similarities.argsort()[::-1]
            similar_indices = [idx for idx in similar_indices if idx != job_idx][:top_n]
            
            return df.iloc[similar_indices]
        
        elif user_profile_text is not None:
            # User profile to job similarity (lazy load BERT model)
            user_embedding = self.bert_model.encode([user_profile_text])
            
            # Calculate similarity with all jobs
            similarities = cosine_similarity(user_embedding, embeddings).flatten()
            
            # Get top similar jobs
            similar_indices = similarities.argsort()[::-1][:top_n]
            
            return df.iloc[similar_indices]
        
        else:
            raise ValueError("Either job_id or user_profile_text must be provided")
    
    def get_collaborative_recommendations(self, user_id, top_n=5) -> pd.DataFrame:
        """Get collaborative filtering recommendations with lazy loading"""
        # Lazy load collaborative model
        similarity_matrix = self.collaborative_model
        
        if similarity_matrix is None:
            logger.warning("Cannot provide collaborative recommendations without user interaction data")
            return pd.DataFrame()
        
        if user_id not in self._user_item_matrix.index:
            logger.warning(f"User {user_id} not found in interaction data")
            return pd.DataFrame()
        
        user_ratings = self._user_item_matrix.loc[user_id]
        rated_jobs = user_ratings[user_ratings > 0].index
        
        predicted_ratings = {}
        
        for job_id in similarity_matrix.columns:
            if job_id not in rated_jobs:
                similar_jobs = similarity_matrix[job_id]
                similar_jobs_rated = similar_jobs[rated_jobs]
                
                if len(similar_jobs_rated) > 0:
                    numerator = sum(similar_jobs_rated * user_ratings[rated_jobs])
                    denominator = sum(abs(similar_jobs_rated))
                    
                    if denominator > 0:
                        predicted_ratings[job_id] = numerator / denominator
                    else:
                        predicted_ratings[job_id] = 0
        
        sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
        top_job_ids = [job_id for job_id, _ in sorted_predictions[:top_n]]
        
        return self.jobs_df[self.jobs_df['Job_ID'].isin(top_job_ids)]
    
    def get_hybrid_recommendations(self, user_id=None, job_id=None, user_profile_text=None, top_n=5) -> pd.DataFrame:
        """
        Get hybrid recommendations with lazy loading of both models
        """
        bert_recommendations = pd.DataFrame()
        collab_recommendations = pd.DataFrame()
        
        # Get BERT-based recommendations (lazy loaded)
        if job_id is not None or user_profile_text is not None:
            bert_recommendations = self.get_bert_recommendations(
                job_id=job_id, 
                user_profile_text=user_profile_text, 
                top_n=top_n
            )
        
        # Get collaborative filtering recommendations (lazy loaded)
        if user_id is not None:
            collab_recommendations = self.get_collaborative_recommendations(user_id, top_n=top_n)
        
        # Combine recommendations
        if not bert_recommendations.empty and not collab_recommendations.empty:
            bert_jobs = set(bert_recommendations['Job_ID'])
            collab_jobs = set(collab_recommendations['Job_ID'])
            
            all_recommended_jobs = bert_jobs.union(collab_jobs)
            
            hybrid_scores = {}
            for job in all_recommended_jobs:
                score = 0
                if job in bert_jobs:
                    rank = list(bert_recommendations['Job_ID']).index(job)
                    score += self.hybrid_weights['content'] * (1 - (rank / len(bert_jobs)))
                
                if job in collab_jobs:
                    rank = list(collab_recommendations['Job_ID']).index(job)
                    score += self.hybrid_weights['collaborative'] * (1 - (rank / len(collab_jobs)))
                
                hybrid_scores[job] = score
            
            sorted_jobs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_job_ids = [job_id for job_id, _ in sorted_jobs]
            
            return self.jobs_df[self.jobs_df['Job_ID'].isin(top_job_ids)]
        
        elif not bert_recommendations.empty:
            return bert_recommendations
        elif not collab_recommendations.empty:
            return collab_recommendations
        else:
            return self.jobs_df.head(top_n)
    
    def recommend_jobs_for_user_profile(self, user_profile: Dict, top_n=5) -> pd.DataFrame:
        """
        Recommend jobs based on user profile using lazy-loaded BERT embeddings
        """
        # Create a comprehensive text description from user profile
        profile_text_parts = []
        
        # Add skills
        if 'skills' in user_profile:
            if isinstance(user_profile['skills'], list):
                skills_text = ' '.join(user_profile['skills'])
            else:
                skills_text = str(user_profile['skills'])
            profile_text_parts.append(f"Skills: {skills_text}")
        
        # Add experience
        if 'experience' in user_profile:
            profile_text_parts.append(f"Experience: {user_profile['experience']} years")
        
        # Add role category
        if 'role_category' in user_profile:
            profile_text_parts.append(f"Role: {user_profile['role_category']}")
        
        # Add industry
        if 'industry' in user_profile:
            profile_text_parts.append(f"Industry: {user_profile['industry']}")
        
        # Add functional area
        if 'functional_area' in user_profile:
            profile_text_parts.append(f"Functional Area: {user_profile['functional_area']}")
        
        # Add job title if specified
        if 'job_title' in user_profile:
            profile_text_parts.append(f"Desired Job Title: {user_profile['job_title']}")
        
        # Combine all parts
        user_profile_text = '. '.join(profile_text_parts)
        
        # Get BERT-based recommendations (lazy loaded)
        return self.get_bert_recommendations(user_profile_text=user_profile_text, top_n=top_n)
    
    def analyze_skill_gaps(self, user_skills: List[str]) -> Dict[str, Any]:
        """Analyze skill gaps based on job market (uses lazy-loaded preprocessed data)"""
        df = self.preprocessed_data
        
        all_skills = []
        for _, job in df.iterrows():
            skills = str(job.get('Key Skills', '')).split(',')
            all_skills.extend([s.strip() for s in skills])
        
        skill_counts = pd.Series(all_skills).value_counts()
        
        skill_gaps = []
        for skill in skill_counts.head(10).index:
            if skill.lower() not in [s.lower() for s in user_skills]:
                skill_gaps.append({
                    'skill': skill,
                    'frequency': skill_counts[skill],
                    'importance': 'high' if skill_counts[skill] > 50 else 'medium'
                })
        
        return {
            'skill_gaps': skill_gaps[:5],
            'match_percentage': 75.0,
            'recommendations_count': len(skill_gaps)
        }
    
    def get_similar_jobs(self, job_id, top_n=5) -> pd.DataFrame:
        """Get jobs similar to a specific job (lazy loaded)"""
        return self.get_bert_recommendations(job_id=job_id, top_n=top_n)
    
    def force_load_all_models(self):
        """Force load all models (useful for pre-warming)"""
        logger.info("Force loading all models...")
        _ = self.bert_model
        _ = self.job_embeddings
        _ = self.collaborative_model
        _ = self.preprocessed_data
        logger.info("All models loaded successfully")
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get the loading status of all models"""
        return {
            'bert_model_loaded': self._bert_model is not None,
            'job_embeddings_loaded': self._job_embeddings is not None,
            'collaborative_model_loaded': self._similarity_matrix is not None,
            'preprocessed_data_loaded': self._preprocessed_data is not None
        }
    
    def clear_cache(self):
        """Clear all cached models to free memory"""
        logger.info("Clearing model cache...")
        self._bert_model = None
        self._job_embeddings = None
        self._similarity_matrix = None
        self._user_item_matrix = None
        self._preprocessed_data = None
        self._cache.clear()
        
        # Clear cached properties
        if hasattr(self, '_cached_bert_model'):
            delattr(self, '_cached_bert_model')
        if hasattr(self, '_cached_job_embeddings'):
            delattr(self, '_cached_job_embeddings')
        if hasattr(self, '_cached_collaborative_model'):
            delattr(self, '_cached_collaborative_model')
        if hasattr(self, '_cached_preprocessed_data'):
            delattr(self, '_cached_preprocessed_data')
        
        logger.info("Model cache cleared")
    
    def save_model(self, filepath: str):
        """Save the model to a file (force loads all models first)"""
        # Force load all models before saving
        self.force_load_all_models()
        
        model_data = {
            'jobs_df': self.jobs_df,
            'user_interactions_df': self.user_interactions_df,
            'job_embeddings': self._job_embeddings,
            'bert_model_name': self.bert_model_name,
            'hybrid_weights': self.hybrid_weights,
            'user_item_matrix': self._user_item_matrix,
            'similarity_matrix': self._similarity_matrix,
            'preprocessed_data': self._preprocessed_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load the model from a file with lazy loading preserved"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance without loading models
        instance = cls(
            jobs_df=model_data['jobs_df'], 
            user_interactions_df=model_data['user_interactions_df'],
            bert_model_name=model_data['bert_model_name']
        )
        
        # Restore pre-computed data (but keep lazy loading for BERT model)
        instance._job_embeddings = model_data['job_embeddings']
        instance.hybrid_weights = model_data['hybrid_weights']
        instance._user_item_matrix = model_data['user_item_matrix']
        instance._similarity_matrix = model_data['similarity_matrix']
        instance._preprocessed_data = model_data['preprocessed_data']
        
        logger.info(f"Model loaded from {filepath} (BERT model will be lazy loaded)")
        return instance


# Enhanced FastAPI Integration with Lazy Loading
class LazyModelManager:
    """Manager for lazy-loaded models in production environments"""
    
    def __init__(self):
        self.model_instance = None
        self.model_lock = threading.Lock()
    
    def get_model(self, jobs_df, user_interactions_df=None, bert_model_name='all-MiniLM-L6-v2'):
        """Get or create model instance with lazy loading"""
        if self.model_instance is None:
            with self.model_lock:
                if self.model_instance is None:
                    self.model_instance = LazyBERTJobRecommenderSystem(
                        jobs_df, user_interactions_df, bert_model_name
                    )
        return self.model_instance
    
    def warm_up_model(self):
        """Pre-warm the model by loading critical components"""
        if self.model_instance:
            # Load only essential components
            _ = self.model_instance.preprocessed_data
            logger.info("Model warmed up with preprocessed data")


def create_lazy_prediction_function(model_manager: LazyModelManager):
    """
    Create a prediction function with lazy loading for FastAPI
    """
    def predict_recommendations(user_data: Dict, num_recommendations: int = 5) -> List[Dict]:
        """
        Enhanced prediction function with lazy loading
        """
        try:
            model = model_manager.model_instance
            if model is None:
                raise ValueError("Model not initialized")
            
            # Get recommendations (models will be lazy loaded as needed)
            recommendations_df = model.recommend_jobs_for_user_profile(
                user_data, 
                top_n=num_recommendations
            )
            
            # Format recommendations for API response
            recommendations = []
            for idx, row in recommendations_df.iterrows():
                recommendations.append({
                    'job_id': int(row['Job_ID']),
                    'job_title': str(row.get('Job Title', 'N/A')),
                    'industry': str(row.get('Industry', 'N/A')),
                    'functional_area': str(row.get('Functional Area', 'N/A')),
                    'experience_required': str(row.get('Job Experience Required', 'N/A')),
                    'key_skills': str(row.get('Key Skills', 'N/A')),
                    'rank': len(recommendations) + 1
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in predict_recommendations: {e}")
            return []
    
    return predict_recommendations


# Example usage with lazy loading
if __name__ == "__main__":
    try:
        # Load existing data
        jobs_df = pd.read_csv('jobs_dataset.csv')  # Update path as needed
        
        # Create mock user interactions
        sample_size = min(500, len(jobs_df))
        job_ids = jobs_df.index[:sample_size].tolist()
        
        sample_interactions = []
        for user_id in range(1, 51):
            num_interactions = np.random.randint(5, 21)
            selected_job_ids = np.random.choice(job_ids, size=min(num_interactions, len(job_ids)), replace=False)
            
            for job_id in selected_job_ids:
                rating = np.random.randint(1, 6)
                sample_interactions.append({
                    'User_ID': user_id,
                    'Job_ID': job_id,
                    'Rating': rating
                })
        
        user_interactions_df = pd.DataFrame(sample_interactions)
        
        # Initialize lazy-loaded recommender
        print("Initializing Lazy BERT Recommender (models not loaded yet)...")
        lazy_recommender = LazyBERTJobRecommenderSystem(jobs_df, user_interactions_df)
        
        # Check model status
        print("Model Status:", lazy_recommender.get_model_status())
        
        # Test recommendations (this will trigger lazy loading)
        print("\nTesting recommendations (will trigger lazy loading)...")
        
        user_profile = {
            'skills': ['Python', 'Data Analysis', 'Machine Learning'],
            'experience': 3,
            'role_category': 'Data Science',
            'industry': 'Technology',
            'functional_area': 'Analytics'
        }
        
        # This will lazy load BERT model and embeddings
        profile_recs = lazy_recommender.recommend_jobs_for_user_profile(user_profile, top_n=3)
        if not profile_recs.empty and 'Job Title' in profile_recs.columns:
            print("Profile-based recommendations:")
            print(profile_recs[['Job Title']].head(3))
        
        # Check model status after first use
        print("\nModel Status after first use:", lazy_recommender.get_model_status())
        
        # Test hybrid recommendations (will lazy load collaborative model)
        hybrid_recs = lazy_recommender.get_hybrid_recommendations(
            user_id=1, 
            user_profile_text="Python developer with machine learning experience", 
            top_n=3
        )
        if not hybrid_recs.empty and 'Job Title' in hybrid_recs.columns:
            print("\nHybrid recommendations:")
            print(hybrid_recs[['Job Title']].head(3))
        
        # Final model status
        print("\nFinal Model Status:", lazy_recommender.get_model_status())
        
        # Save the model
        lazy_recommender.save_model('lazy_bert_job_recommender.pkl')
        
        # Demonstrate memory management
        print("\nClearing cache...")
        lazy_recommender.clear_cache()
        print("Model Status after cache clear:", lazy_recommender.get_model_status())
        
        # Create production-ready model manager
        model_manager = LazyModelManager()
        model_manager.model_instance = lazy_recommender
        
        # Create prediction function
        predict_func = create_lazy_prediction_function(model_manager)
        
        # Test prediction function
        test_user_data = {
            'skills': ['Python', 'Machine Learning'],
            'experience': 2,
            'role_category': 'Data Science'
        }
        
        predictions = predict_func(test_user_data, 3)
        print(f"\nAPI-compatible predictions: {len(predictions)} recommendations generated")
        
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
        import traceback
        traceback.print_exc()