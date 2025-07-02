#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class JobRecommenderSystem:
    def __init__(self, jobs_df, user_interactions_df=None):
        """
        Initialize the Job Recommender System with job data and optional user interactions

        Parameters:
        jobs_df (DataFrame): DataFrame containing job listings with columns:
                           'Job Salary', 'Job Experience Required', 'Key Skills', 
                           'Role Category', 'Functional Area', 'Industry', 'Job Title'
        user_interactions_df (DataFrame): Optional DataFrame containing user-job interactions
                                        with columns: 'User_ID', 'Job_ID', 'Rating'
        """
        self.jobs_df = jobs_df.copy()
        # Adding a job ID if not present
        if 'Job_ID' not in self.jobs_df.columns:
            self.jobs_df['Job_ID'] = range(len(self.jobs_df))

        self.user_interactions_df = user_interactions_df
        self.content_based_model = None
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.tfidf_matrix = None
        self.hybrid_weights = {'content': 0.5, 'collaborative': 0.5}

    def preprocess_data(self):
        """Preprocess job data for recommendation"""
        # Handle salary - convert to numeric, or create a binary indicator
        if 'Job Salary' in self.jobs_df.columns:
            # Try to extract numeric values from salary
            try:
                # Try direct conversion to numeric
                self.jobs_df['Salary_Numeric'] = pd.to_numeric(self.jobs_df['Job Salary'], errors='coerce')

                # If all values are NaN after conversion, using a binary indicator instead
                if self.jobs_df['Salary_Numeric'].isna().all():
                    print("Warning: Could not convert any salary values to numeric. Using binary indicator instead.")
                    self.jobs_df['Salary_Disclosed'] = self.jobs_df['Job Salary'].apply(
                        lambda x: 0 if str(x).lower().strip() in ['not disclosed', 'not disclosed by recruiter', 'na', 'n/a', ''] else 1
                    )
                else:
                    # Filled NaNs with median of valid values
                    median_salary = self.jobs_df['Salary_Numeric'].median()
                    self.jobs_df['Salary_Numeric'] = self.jobs_df['Salary_Numeric'].fillna(median_salary)

                    # Normalize the numeric salary
                    scaler = MinMaxScaler()
                    self.jobs_df['Normalized_Salary'] = scaler.fit_transform(self.jobs_df[['Salary_Numeric']])
            except Exception as e:
                print(f"Warning: Error processing salary data - {e}. Using binary indicator instead.")
                self.jobs_df['Salary_Disclosed'] = self.jobs_df['Job Salary'].apply(
                    lambda x: 0 if str(x).lower().strip() in ['not disclosed', 'not disclosed by recruiter', 'na', 'n/a', ''] else 1
                )

        # Handling experience - convert to numeric if possible
        if 'Job Experience Required' in self.jobs_df.columns:
            try:
                # Extract numeric years from experience text (e.g., "5 - 10 yrs" -> 7.5)
                def extract_years(exp_text):
                    if pd.isna(exp_text) or not isinstance(exp_text, str):
                        return 0

                    # Try to find patterns like "X - Y yrs" or "X yrs"
                    exp_text = exp_text.lower().strip()
                    if '-' in exp_text:
                        parts = exp_text.split('-')
                        if len(parts) == 2:
                            try:
                                min_exp = float(''.join(c for c in parts[0] if c.isdigit() or c == '.'))
                                max_exp = float(''.join(c for c in parts[1] if c.isdigit() or c == '.'))
                                return (min_exp + max_exp) / 2  # Average of min and max
                            except ValueError:
                                return 0
                    else:
                        # Try to extract a single number
                        try:
                            return float(''.join(c for c in exp_text if c.isdigit() or c == '.'))
                        except ValueError:
                            return 0

                self.jobs_df['Experience_Numeric'] = self.jobs_df['Job Experience Required'].apply(extract_years)

                # Normalize the numeric experience
                scaler = MinMaxScaler()
                self.jobs_df['Normalized_Experience'] = scaler.fit_transform(self.jobs_df[['Experience_Numeric']])
            except Exception as e:
                print(f"Warning: Error processing experience data - {e}")

        # Created a combined text feature for TF-IDF
        text_features = []

        # Added all text features that exist in the dataframe
        if 'Job Title' in self.jobs_df.columns:
            text_features.append(self.jobs_df['Job Title'].fillna('').astype(str))

        if 'Key Skills' in self.jobs_df.columns:
            text_features.append(self.jobs_df['Key Skills'].fillna('').astype(str))

        if 'Role Category' in self.jobs_df.columns:
            text_features.append(self.jobs_df['Role Category'].fillna('').astype(str))

        if 'Functional Area' in self.jobs_df.columns:
            text_features.append(self.jobs_df['Functional Area'].fillna('').astype(str))

        if 'Industry' in self.jobs_df.columns:
            text_features.append(self.jobs_df['Industry'].fillna('').astype(str))

        # Combine all text features
        if text_features:
            self.jobs_df['Combined_Features'] = pd.Series(' '.join(str(val) for val in vals) 
                                                for vals in zip(*text_features))
        else:
            print("Warning: No text features found for content-based filtering")
            self.jobs_df['Combined_Features'] = ""

    def build_content_based_model(self):
        """Build the content-based filtering model"""
        self.preprocess_data()

        # Create TF-IDF vectors for job descriptions
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.jobs_df['Combined_Features'])

        self.content_based_model = {
            'tfidf_matrix': self.tfidf_matrix,
            'job_indices': {idx: job_id for idx, job_id in enumerate(self.jobs_df['Job_ID'])}
        }

        return self.content_based_model

    def build_collaborative_model(self):
        """
        Build collaborative filtering model using user-item matrix and item-item similarity
        """
        if self.user_interactions_df is None:
            print("No user interaction data available for collaborative filtering")
            return None

        # Create user-item matrix
        # Convert ratings to a user-item matrix
        user_item_matrix = pd.pivot_table(
            self.user_interactions_df,
            values='Rating',
            index='User_ID',
            columns='Job_ID',
            fill_value=0
        )

        # Store the user-item matrix
        self.user_item_matrix = user_item_matrix

        # Calculate item-item similarity matrix using cosine similarity
        # Transpose the matrix to get item-item similarity
        item_item_similarity = cosine_similarity(user_item_matrix.T)
        self.similarity_matrix = pd.DataFrame(
            item_item_similarity,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns
        )

        return self.similarity_matrix

    def get_content_based_recommendations(self, job_id, top_n=5):
        """Get content-based job recommendations for a given job ID"""
        if self.content_based_model is None:
            self.build_content_based_model()

        # Check if job_id exists in our data
        if job_id not in self.jobs_df['Job_ID'].values:
            print(f"Job ID {job_id} not found in the dataset")
            return pd.DataFrame()

        # Find the index of the job in our dataset
        job_idx = self.jobs_df[self.jobs_df['Job_ID'] == job_id].index[0]

        # Calculate cosine similarity between this job and all others
        cosine_similarities = cosine_similarity(self.tfidf_matrix[job_idx:job_idx+1], self.tfidf_matrix).flatten()

        # Get the indices of the top N most similar jobs (excluding the input job)
        similar_indices = cosine_similarities.argsort()[::-1]
        similar_indices = [idx for idx in similar_indices if idx != job_idx][:top_n]

        # Return the similar jobs
        return self.jobs_df.iloc[similar_indices]

    def get_collaborative_recommendations(self, user_id, top_n=5):
        """
        Get collaborative filtering recommendations for a user using item-based CF
        """
        if self.similarity_matrix is None:
            if self.user_interactions_df is None:
                print("Cannot provide collaborative recommendations without user interaction data")
                return pd.DataFrame()
            self.build_collaborative_model()

        # Check if user exists in the user-item matrix
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found in interaction data")
            return pd.DataFrame()

        # Get the user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]

        # Jobs the user has already rated
        rated_jobs = user_ratings[user_ratings > 0].index

        # Initialize a dictionary to store the predicted ratings
        predicted_ratings = {}

        # For each job the user hasn't rated
        for job_id in self.similarity_matrix.columns:
            if job_id not in rated_jobs:
                # Get similar jobs that the user has rated
                similar_jobs = self.similarity_matrix[job_id]
                similar_jobs_rated = similar_jobs[rated_jobs]

                # If there are similar jobs the user has rated
                if len(similar_jobs_rated) > 0:
                    # Calculate the weighted average rating
                    numerator = sum(similar_jobs_rated * user_ratings[rated_jobs])
                    denominator = sum(abs(similar_jobs_rated))

                    if denominator > 0:
                        predicted_ratings[job_id] = numerator / denominator
                    else:
                        predicted_ratings[job_id] = 0

        # Sort jobs by predicted rating
        sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

        # Get top N job IDs
        top_job_ids = [job_id for job_id, _ in sorted_predictions[:top_n]]

        # Return the top N recommended jobs
        return self.jobs_df[self.jobs_df['Job_ID'].isin(top_job_ids)]

    def matrix_factorization(self, R, P=None, Q=None, K=10, steps=50, alpha=0.01, beta=0.02):
        """
        Matrix Factorization using Gradient Descent
        Parameters:
        R (ndarray): User-item rating matrix
        P (ndarray): User features matrix
        Q (ndarray): Item features matrix
        K (int): Number of latent features
        steps (int): Number of iterations
        alpha (float): Learning rate
        beta (float): Regularization parameter

        Returns:
        P (ndarray): Updated user features matrix
        Q (ndarray): Updated item features matrix
        """
        # Get dimensions of the matrix
        M, N = R.shape

        # Initialize P and Q if not provided
        if P is None:
            P = np.random.rand(M, K)  # User features
        if Q is None:
            Q = np.random.rand(N, K)  # Item features

        # Create a mask for non-zero entries
        mask = (R > 0).astype(float)

        # Matrix factorization using Gradient Descent
        for step in range(steps):
            for i in range(M):
                for j in range(N):
                    if mask[i, j] > 0:  # Only for non-zero entries
                        # Compute error
                        eij = R[i, j] - np.dot(P[i,:], Q[j,:].T)

                        # Update P and Q
                        for k in range(K):
                            P[i, k] += alpha * (2 * eij * Q[j, k] - beta * P[i, k])
                            Q[j, k] += alpha * (2 * eij * P[i, k] - beta * Q[j, k])

            # Compute current RMSE
            error = 0
            count = 0
            for i in range(M):
                for j in range(N):
                    if mask[i, j] > 0:
                        error += (R[i, j] - np.dot(P[i,:], Q[j,:].T)) ** 2
                        count += 1
            rmse = np.sqrt(error / count) if count > 0 else 0

            # Early stopping if error is small enough
            if rmse < 0.001:
                break

        return P, Q

    def build_mf_collaborative_model(self, k=10, steps=50):
        """
        Build collaborative filtering model using matrix factorization
        """
        if self.user_interactions_df is None:
            print("No user interaction data available for collaborative filtering")
            return None

        # Create user-item matrix
        user_item_matrix = pd.pivot_table(
            self.user_interactions_df,
            values='Rating',
            index='User_ID',
            columns='Job_ID',
            fill_value=0
        )

        # Store the user-item matrix and indices
        self.user_item_matrix = user_item_matrix
        self.user_indices = {i: user_id for i, user_id in enumerate(user_item_matrix.index)}
        self.job_indices = {i: job_id for i, job_id in enumerate(user_item_matrix.columns)}
        self.reverse_user_indices = {user_id: i for i, user_id in self.user_indices.items()}
        self.reverse_job_indices = {job_id: i for i, job_id in self.job_indices.items()}

        # Matrix factorization
        R = user_item_matrix.values
        self.P, self.Q = self.matrix_factorization(R, K=k, steps=steps)

        return self.P, self.Q

    def get_mf_recommendations(self, user_id, top_n=5):
        """
        Get collaborative filtering recommendations using matrix factorization
        """
        if not hasattr(self, 'P') or not hasattr(self, 'Q'):
            self.build_mf_collaborative_model()

        # Check if user exists
        if user_id not in self.reverse_user_indices:
            print(f"User {user_id} not found in interaction data")
            return pd.DataFrame()

        # Get user index
        user_idx = self.reverse_user_indices[user_id]

        # Get user's actual ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_jobs = user_ratings[user_ratings > 0].index.tolist()

        # Predict ratings for all jobs
        user_features = self.P[user_idx, :]
        predicted_ratings = {}

        for job_idx, job_id in self.job_indices.items():
            if job_id not in rated_jobs:  # Only recommend jobs the user hasn't rated
                job_features = self.Q[job_idx, :]
                predicted_rating = np.dot(user_features, job_features.T)
                predicted_ratings[job_id] = predicted_rating

        # Sort jobs by predicted rating
        sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

        # Get top N job IDs
        top_job_ids = [job_id for job_id, _ in sorted_predictions[:top_n]]

        # Return the top N recommended jobs
        return self.jobs_df[self.jobs_df['Job_ID'].isin(top_job_ids)]

    def get_hybrid_recommendations(self, user_id, job_id=None, top_n=5, mf=False):
        """
        Get hybrid recommendations using both content-based and collaborative filtering

        Parameters:
        user_id: User ID for collaborative filtering
        job_id: Optional job ID for content-based filtering (e.g., job the user is currently viewing)
        top_n: Number of recommendations to return
        mf: Whether to use matrix factorization for collaborative filtering
        """
        content_recommendations = pd.DataFrame()
        collab_recommendations = pd.DataFrame()

        # Get content-based recommendations if job_id is provided
        if job_id is not None:
            content_recommendations = self.get_content_based_recommendations(job_id, top_n=top_n)

        # Get collaborative filtering recommendations
        if self.user_interactions_df is not None:
            if mf:
                collab_recommendations = self.get_mf_recommendations(user_id, top_n=top_n)
            else:
                collab_recommendations = self.get_collaborative_recommendations(user_id, top_n=top_n)

        # If we have both types of recommendations, combine them with weights
        if not content_recommendations.empty and not collab_recommendations.empty:
            # Create a score for each recommendation type
            content_jobs = set(content_recommendations['Job_ID'])
            collab_jobs = set(collab_recommendations['Job_ID'])

            # Combine the job sets
            all_recommended_jobs = content_jobs.union(collab_jobs)

            # Calculate hybrid scores
            hybrid_scores = {}
            for job in all_recommended_jobs:
                score = 0
                if job in content_jobs:
                    # Give more weight to higher-ranked jobs in content recommendations
                    rank = list(content_recommendations['Job_ID']).index(job)
                    score += self.hybrid_weights['content'] * (1 - (rank / len(content_jobs)))

                if job in collab_jobs:
                    # Give more weight to higher-ranked jobs in collaborative recommendations
                    rank = list(collab_recommendations['Job_ID']).index(job)
                    score += self.hybrid_weights['collaborative'] * (1 - (rank / len(collab_jobs)))

                hybrid_scores[job] = score

            # Sort by hybrid score and get top_n
            sorted_jobs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_job_ids = [job_id for job_id, _ in sorted_jobs]

            return self.jobs_df[self.jobs_df['Job_ID'].isin(top_job_ids)]

        # If we only have one type of recommendations, return that
        elif not content_recommendations.empty:
            return content_recommendations
        elif not collab_recommendations.empty:
            return collab_recommendations
        else:
            # If we have no recommendations, return top jobs by some other metric
            # Try using salary if available, otherwise just return first top_n jobs
            if 'Normalized_Salary' in self.jobs_df.columns:
                return self.jobs_df.sort_values('Normalized_Salary', ascending=False).head(top_n)
            else:
                return self.jobs_df.head(top_n)

    def recommend_jobs_for_user_profile(self, user_profile, top_n=5):
        """
        Recommend jobs based on user profile (skills, experience, etc.)

        Parameters:
        user_profile: Dictionary containing user profile information with keys like
                      'Skills', 'Experience', 'Role Category', 'Industry', etc.
        top_n: Number of recommendations to return
        """
        if self.content_based_model is None:
            self.build_content_based_model()

        # Create a pseudo-job entry from the user profile
        pseudo_job = {
            'Job Title': user_profile.get('Desired Job Title', ''),
            'Key Skills': user_profile.get('Skills', ''),
            'Role Category': user_profile.get('Role Category', ''),
            'Functional Area': user_profile.get('Functional Area', ''),
            'Industry': user_profile.get('Industry', ''),
            'Job Experience Required': user_profile.get('Experience', 0),
            'Job Salary': user_profile.get('Expected Salary', 0)
        }

        # Combine text features as we did for the dataset
        text_features = [
            str(pseudo_job['Job Title']),
            str(pseudo_job['Key Skills']),
            str(pseudo_job['Role Category']),
            str(pseudo_job['Functional Area']),
            str(pseudo_job['Industry'])
        ]

        pseudo_job_features = ' '.join(text_features)

        # Create TF-IDF vector for the pseudo job
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit(self.jobs_df['Combined_Features'])
        jobs_tfidf = tfidf.transform(self.jobs_df['Combined_Features'])
        pseudo_job_tfidf = tfidf.transform([pseudo_job_features])

        # Calculate cosine similarity with all jobs
        cosine_similarities = cosine_similarity(pseudo_job_tfidf, jobs_tfidf).flatten()

        # Get indices of top N similar jobs
        similar_indices = cosine_similarities.argsort()[::-1][:top_n]

        # Return the similar jobs
        return self.jobs_df.iloc[similar_indices]



# # Example usage with mock user interactions
# if __name__ == "__main__":
#     # Load your dataset
#     try:
#         jobs_df = pd.read_csv('Downloads/jobs.csv')

#         # Display some info about the dataset
#         print(f"Dataset loaded with {len(jobs_df)} jobs")
#         print(f"Columns: {jobs_df.columns.tolist()}")

#         # Check data types for key columns
#         for col in ['Job Salary', 'Job Experience Required', 'Key Skills']:
#             if col in jobs_df.columns:
#                 print(f"{col} data type: {jobs_df[col].dtype}")
#                 print(f"{col} sample values: {jobs_df[col].head(3).tolist()}")

#         # Create a mock user interactions dataset
#         print("Creating a mock user interactions dataset...")

#         # Ensure jobs have Job_ID
#         if 'Job_ID' not in jobs_df.columns:
#             jobs_df['Job_ID'] = range(len(jobs_df))

#         # Create a sample of job IDs to use in interactions (limit to 500 for efficiency)
#         sample_size = min(500, len(jobs_df))
#         job_ids = jobs_df['Job_ID'].iloc[:sample_size].tolist()

#         # Create 50 mock users
#         num_users = 50
#         user_ids = list(range(1, num_users + 1))

#         # Generate random interactions
#         sample_interactions = []
#         for user_id in user_ids:
#             # Each user interacts with 5-20 random jobs
#             num_interactions = np.random.randint(5, 21)
#             selected_job_ids = np.random.choice(job_ids, size=min(num_interactions, len(job_ids)), replace=False)

#             for job_id in selected_job_ids:
#                 # Generate a random rating between 1 and 5
#                 rating = np.random.randint(1, 6)
#                 sample_interactions.append({
#                     'User_ID': user_id,
#                     'Job_ID': job_id,
#                     'Rating': rating
#                 })

#         user_interactions_df = pd.DataFrame(sample_interactions)
#         print(f"Created mock user interactions with {len(user_interactions_df)} records")

#         # Initialize the recommender system
#         recommender = JobRecommenderSystem(jobs_df, user_interactions_df)

#         # Build models
#         print("Building content-based model...")
#         recommender.build_content_based_model()

#         print("Building collaborative filtering model...")
#         recommender.build_collaborative_model()

#         # Alternatively, use matrix factorization (smaller number of steps for demo)
#         print("Building matrix factorization model...")
#         recommender.build_mf_collaborative_model(k=10, steps=20)

#         # Example of how to get recommendations
#         # Use an actual job_id from the dataset
#         sample_job_id = jobs_df['Job_ID'].iloc[0]

#         # Use an actual user_id from the interactions
#         sample_user_id = user_interactions_df['User_ID'].iloc[0]

#         print("\nEXAMPLE RECOMMENDATIONS:")
#         print(f"Content-based recommendations for job {sample_job_id}:")
#         content_recs = recommender.get_content_based_recommendations(sample_job_id, top_n=3)
#         if not content_recs.empty:
#             if 'Job Title' in content_recs.columns:
#                 print(content_recs[['Job_ID', 'Job Title']].head(3))
#             else:
#                 print(content_recs[['Job_ID']].head(3))
#         else:
#             print("No content-based recommendations found")

#         print(f"\nCollaborative filtering recommendations for user {sample_user_id}:")
#         collab_recs = recommender.get_collaborative_recommendations(sample_user_id, top_n=3)
#         if not collab_recs.empty:
#             if 'Job Title' in collab_recs.columns:
#                 print(collab_recs[['Job_ID', 'Job Title']].head(3))
#             else:
#                 print(collab_recs[['Job_ID']].head(3))
#         else:
#             print("No collaborative filtering recommendations found")

#         print(f"\nMatrix factorization recommendations for user {sample_user_id}:")
#         mf_recs = recommender.get_mf_recommendations(sample_user_id, top_n=3)
#         if not mf_recs.empty:
#             if 'Job Title' in mf_recs.columns:
#                 print(mf_recs[['Job_ID', 'Job Title']].head(3))
#             else:
#                 print(mf_recs[['Job_ID']].head(3))
#         else:
#             print("No matrix factorization recommendations found")

#         print("\nHybrid recommendations:")
#         hybrid_recs = recommender.get_hybrid_recommendations(sample_user_id, sample_job_id, top_n=3)
#         if not hybrid_recs.empty:
#             if 'Job Title' in hybrid_recs.columns:
#                 print(hybrid_recs[['Job_ID', 'Job Title']].head(3))
#             else:
#                 print(hybrid_recs[['Job_ID']].head(3))
#         else:
#             print("No hybrid recommendations found")

#         # User profile example
#         user_profile = {
#             'Skills': 'Python, Data Analysis, Machine Learning',
#             'Experience': 3,
#             'Role Category': 'Data Science',
#             'Industry': 'Technology',
#             'Functional Area': 'Analytics'
#         }
#         print("\nProfile-based recommendations:")
#         profile_recs = recommender.recommend_jobs_for_user_profile(user_profile, top_n=3)
#         if not profile_recs.empty:
#             if 'Job Title' in profile_recs.columns:
#                 print(profile_recs[['Job_ID', 'Job Title']].head(3))
#             else:
#                 print(profile_recs[['Job_ID']].head(3))
#         else:
#             print("No profile-based recommendations found")

#     except Exception as e:
#         import traceback
#         print(f"Error in job recommender: {e}")
#         traceback.print_exc()


# # In[ ]:




