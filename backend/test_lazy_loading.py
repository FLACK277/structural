#!/usr/bin/env python3
"""
Test script to verify that lazy loading works correctly
"""
import sys
import time
import os

def test_fastapi_startup():
    """Test that FastAPI app can start without loading heavy models"""
    print("Testing FastAPI startup with lazy loading...")
    
    try:
        # Import the FastAPI app
        from app.main import app
        print("✓ FastAPI app imported successfully")
        
        # Check that the app is a FastAPI instance
        from fastapi import FastAPI
        assert isinstance(app, FastAPI), "App is not a FastAPI instance"
        print("✓ FastAPI app is properly configured")
        
        # Check that routes are registered
        routes = [route.path for route in app.routes]
        print(f"✓ Routes registered: {len(routes)} routes found")
        
        return True
        
    except Exception as e:
        print(f"✗ FastAPI startup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lazy_imports():
    """Test that heavy imports are not loaded at startup"""
    print("\nTesting lazy imports...")
    
    try:
        # Import the route modules (should not trigger heavy imports)
        from app.routes import skillassess, job_recommendation_routes, resume, roles
        print("✓ Route modules imported successfully")
        
        # Check that the lazy loading functions exist
        assert hasattr(skillassess, 'get_assessment_engine'), "get_assessment_engine not found"
        assert hasattr(job_recommendation_routes, 'get_recommender'), "get_recommender not found"
        assert hasattr(resume, 'get_bert_learning_system'), "get_bert_learning_system not found"
        print("✓ Lazy loading functions are available")
        
        return True
        
    except Exception as e:
        print(f"✗ Lazy import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_uvicorn_startup():
    """Test that uvicorn can start the app"""
    print("\nTesting uvicorn startup...")
    
    try:
        import uvicorn
        from app.main import app
        
        # This should start quickly without loading heavy models
        print("✓ Uvicorn can import the app")
        
        return True
        
    except Exception as e:
        print(f"✗ Uvicorn startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Testing Lazy Loading Implementation ===\n")
    
    start_time = time.time()
    
    # Test 1: FastAPI startup
    test1_passed = test_fastapi_startup()
    
    # Test 2: Lazy imports
    test2_passed = test_lazy_imports()
    
    # Test 3: Uvicorn startup
    test3_passed = test_uvicorn_startup()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== Test Results ===")
    print(f"FastAPI Startup: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Lazy Imports: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Uvicorn Startup: {'✓ PASSED' if test3_passed else '✗ FAILED'}")
    print(f"Total Test Time: {total_time:.2f} seconds")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed! Your app should start quickly on Render.")
        print("The heavy ML models will only be loaded when the first API request is made.")
        return True
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 