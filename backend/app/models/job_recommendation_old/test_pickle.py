import pickle
import pandas as pd
import os
import traceback

print("Starting test_pickle.py...")

def test_pickle_loading():
    """Test if the pickle file loads correctly"""
    try:
        # Try to load the pickle file
        with open('job_recommender_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print("✅ Pickle file loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Test with a sample user profile
        user_profile = {
            'Skills': 'Python, Data Analysis, Machine Learning',
            'Experience': 3,
            'Role Category': 'Data Science',
            'Industry': 'Technology',
            'Functional Area': 'Analytics',
            'Desired Job Title': 'Data Scientist'
        }
        
        print("\n🧪 Testing with sample user profile:")
        print(f"User Profile: {user_profile}")
        
        # Get recommendations
        recommendations = model.recommend_jobs_for_user_profile(user_profile, top_n=3)
        
        if not recommendations.empty:
            print("\n✅ Recommendations generated successfully!")
            print(f"Number of recommendations: {len(recommendations)}")
            
            # Display recommendations
            for idx, row in recommendations.iterrows():
                print(f"\nJob {row.get('Job_ID', 'N/A')}:")
                print(f"  Title: {row.get('Job Title', 'N/A')}")
                print(f"  Industry: {row.get('Industry', 'N/A')}")
                print(f"  Skills: {row.get('Key Skills', 'N/A')[:100]}...")
        else:
            print("\n❌ No recommendations generated")
            
        return True, model
        
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    test_pickle_loading() 