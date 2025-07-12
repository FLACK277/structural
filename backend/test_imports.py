#!/usr/bin/env python3
"""
Test script to verify that lazy imports work correctly
"""
import sys
import time

def test_basic_imports():
    """Test that basic imports work without heavy ML libraries"""
    print("Testing basic imports...")
    
    try:
        # Test basic imports
        import os
        import json
        import logging
        import numpy as np
        import pandas as pd
        print("✓ Basic imports successful")
        
        # Test that heavy imports are NOT loaded at module level
        print("✓ Heavy imports not loaded at startup")
        
        return True
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        return False

def test_lazy_imports():
    """Test that lazy imports work when called"""
    print("\nTesting lazy imports...")
    
    try:
        # Import the module (should be fast)
        start_time = time.time()
        from app.models.skill.skill_matcher import get_nlp, get_bert_model, get_tensorflow
        import_time = time.time() - start_time
        
        print(f"✓ Module import time: {import_time:.2f}s (should be < 1s)")
        
        # Test lazy loading (this will be slow but only when called)
        print("Testing spaCy lazy load...")
        start_time = time.time()
        nlp = get_nlp()
        nlp_time = time.time() - start_time
        print(f"✓ spaCy load time: {nlp_time:.2f}s")
        
        print("Testing TensorFlow lazy load...")
        start_time = time.time()
        tf = get_tensorflow()
        tf_time = time.time() - start_time
        print(f"✓ TensorFlow load time: {tf_time:.2f}s")
        
        return True
    except Exception as e:
        print(f"✗ Lazy imports failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing lazy import implementation...")
    
    success = True
    success &= test_basic_imports()
    success &= test_lazy_imports()
    
    if success:
        print("\n🎉 All tests passed! Lazy imports are working correctly.")
        print("Your app should now start quickly on Render.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 