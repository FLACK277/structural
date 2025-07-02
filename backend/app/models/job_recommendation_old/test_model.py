import pickle
from job_recommendation import JobRecommenderSystem  # Must match class in pkl

# Load the model
with open("job_recommender_model.pkl", "rb") as f:
    recommender = pickle.load(f)

# Prepare sample input — depends on what the class expects
user_input = {
    "skills": ["Python", "Machine Learning", "Data Analysis"],
    "experience": 2,
    "preferred_role": "Data Scientist"
}

# Call the recommend method
result = recommender.recommend_jobs(user_input)
print("Recommended Jobs:", result)
