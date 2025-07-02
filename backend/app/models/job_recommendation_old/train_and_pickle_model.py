import pandas as pd
import numpy as np
import pickle
import os
from app.models.job_recommendation import JobRecommenderSystem

# 1. Load the jobs dataset
jobs_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'jobs_dataset.csv'))

# 2. Create mock user interactions (as in your original script)
if 'Job_ID' not in jobs_df.columns:
    jobs_df['Job_ID'] = range(len(jobs_df))

sample_size = min(500, len(jobs_df))
job_ids = jobs_df['Job_ID'].iloc[:sample_size].tolist()
num_users = 50
user_ids = list(range(1, num_users + 1))
sample_interactions = []
for user_id in user_ids:
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

# 3. Train the recommender
recommender = JobRecommenderSystem(jobs_df, user_interactions_df)
recommender.build_content_based_model()
recommender.build_collaborative_model()
recommender.build_mf_collaborative_model(k=10, steps=20)

# 4. Pickle the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'job_recommender_model.pkl')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(recommender, f)

print('✅ Model trained and pickled as job_recommender_model.pkl') 