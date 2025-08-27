#!/usr/bin/env python3
"""
Test script for the enhanced Multi-Agent AI Bidding System.
This script tests the core components to ensure everything is working correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_enhanced_agents():
    """Test the enhanced bidding agents."""
    print("🧪 Testing Enhanced Bidding Agents...")
    
    try:
        from src.agents.enhanced_bidding_agent import EnhancedBiddingAgent
        
        # Create test agents
        agent1 = EnhancedBiddingAgent("Test_Agent_1", "aggressive")
        agent2 = EnhancedBiddingAgent("Test_Agent_2", "balanced")
        agent3 = EnhancedBiddingAgent("Test_Agent_3", "conservative")
        
        # Test bid generation
        market_threshold = 500
        rounds_remaining = 10
        competitor_bids = [400, 450, 480]
        
        bid1 = agent1.generate_bid(market_threshold, rounds_remaining, competitor_bids)
        bid2 = agent2.generate_bid(market_threshold, rounds_remaining, competitor_bids)
        bid3 = agent3.generate_bid(market_threshold, rounds_remaining, competitor_bids)
        
        print(f"✅ Agent 1 (Aggressive) bid: {bid1:.2f}")
        print(f"✅ Agent 2 (Balanced) bid: {bid2:.2f}")
        print(f"✅ Agent 3 (Conservative) bid: {bid3:.2f}")
        
        # Test reward update
        agent1.update_reward(10)  # Win
        agent2.update_reward(-1)  # Loss
        
        metrics1 = agent1.get_performance_metrics()
        print(f"✅ Agent 1 metrics: {metrics1}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced agents test failed: {e}")
        return False

def test_recommendation_engine():
    """Test the recommendation engine."""
    print("\n🧪 Testing Recommendation Engine...")
    
    try:
        from src.core.recommendation_engine import OptimalBidRecommendationEngine
        
        # Create test data
        test_data = pd.DataFrame({
            'Round': [1, 1, 1, 2, 2, 2],
            'Agent': ['Agent1', 'Agent2', 'Agent3', 'Agent1', 'Agent2', 'Agent3'],
            'Bid': [400, 450, 480, 380, 420, 460],
            'Winning_Bid': [1, 0, 0, 1, 0, 0]
        })
        
        # Create recommendation engine
        engine = OptimalBidRecommendationEngine()
        
        # Test recommendation generation
        recommendation = engine.analyze_simulation_results(
            test_data, 
            market_threshold=500,
            user_agent_name="Agent1"
        )
        
        print(f"✅ Recommended bid: {recommendation.recommended_bid:.2f}")
        print(f"✅ Win probability: {recommendation.win_probability:.1%}")
        print(f"✅ Confidence score: {recommendation.confidence_score:.1%}")
        print(f"✅ Risk level: {recommendation.risk_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Recommendation engine test failed: {e}")
        return False

def test_enhanced_simulation():
    """Test the enhanced simulation."""
    print("\n🧪 Testing Enhanced Simulation...")
    
    try:
        from src.core.enhanced_bidding_simulation import EnhancedBiddingSimulation
        from src.agents.enhanced_bidding_agent import EnhancedBiddingAgent
        
        # Create test agents
        agents = [
            EnhancedBiddingAgent("Test_Agent_1", "aggressive"),
            EnhancedBiddingAgent("Test_Agent_2", "balanced"),
            EnhancedBiddingAgent("Test_Agent_3", "conservative")
        ]
        
        # Create simulation with cost price
        simulation = EnhancedBiddingSimulation(
            agents=agents,
            rounds=5,
            initial_threshold=500,
            cost_price=300
        )
        
        # Run simulation
        results = simulation.run_simulation()
        
        print(f"✅ Simulation completed with {len(results)} data points")
        print(f"✅ Results shape: {results.shape}")
        print(f"✅ Columns: {list(results.columns)}")
        
        # Print performance summary
        simulation.print_performance_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced simulation test failed: {e}")
        return False

def test_visualization_components():
    """Test the visualization components."""
    print("\n🧪 Testing Visualization Components...")
    
    try:
        from frontend.components.enhanced_visualization import EnhancedVisualization
        
        # Create test data
        test_data = pd.DataFrame({
            'Round': [1, 1, 1, 2, 2, 2],
            'Agent': ['Agent1', 'Agent2', 'Agent3', 'Agent1', 'Agent2', 'Agent3'],
            'Bid': [400, 450, 480, 380, 420, 460],
            'Winning_Bid': [1, 0, 0, 1, 0, 0]
        })
        
        # Create visualization component
        viz = EnhancedVisualization()
        
        # Test executive summary creation
        summary = viz.create_executive_summary(test_data)
        print("✅ Executive summary created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization components test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced Multi-Agent AI Bidding System")
    print("=" * 60)
    
    tests = [
        test_enhanced_agents,
        test_recommendation_engine,
        test_enhanced_simulation,
        test_visualization_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
