#!/usr/bin/env python3
"""
Deployment test script to verify all dependencies are working correctly.
"""

def test_imports():
    """Test all critical imports for the enhanced app."""
    try:
        print("Testing imports...")
        
        # Core dependencies
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
        
        import torch
        print("✅ PyTorch imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        import seaborn as sns
        print("✅ Seaborn imported successfully")
        
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported successfully")
        
        import openai
        print("✅ OpenAI imported successfully")
        
        from dotenv import load_dotenv
        print("✅ Python-dotenv imported successfully")
        
        # App-specific imports
        from frontend.components.enhanced_visualization import EnhancedVisualization
        print("✅ EnhancedVisualization imported successfully")
        
        from src.agents.enhanced_bidding_agent import EnhancedBiddingAgent
        print("✅ EnhancedBiddingAgent imported successfully")
        
        from src.core.recommendation_engine import OptimalBidRecommendationEngine
        print("✅ OptimalBidRecommendationEngine imported successfully")
        
        print("\n🎉 All imports successful! Deployment should work correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
