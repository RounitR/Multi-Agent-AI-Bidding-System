import sys
import os
import torch
import argparse
import openai
from dotenv import load_dotenv

# Ensure the src module is available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.agents.enhanced_bidding_agent import EnhancedBiddingAgent
from src.core.enhanced_bidding_simulation import EnhancedBiddingSimulation
from src.utils.data_handler import DataHandler

# Load OpenAI API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def check_openai_api():
    """Checks if OpenAI API is working before running the simulation."""
    if not OPENAI_API_KEY:
        print("No OpenAI API key found. AI-enhanced bidding will be disabled.")
        return False
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Test OpenAI connection"}]
        )
        print("✅ OpenAI API Key is working!")
        return True
    except Exception as e:
        print(f"⚠️ OpenAI API Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run AI-powered Multi-Agent Bidding Simulation")
    parser.add_argument("--visualize", action="store_true", help="Visualize bid trends after simulation")
    args = parser.parse_args()

    # Check if OpenAI API is working
    ai_enabled = check_openai_api()

    # Initialize enhanced bidding agents
    agents = [
        EnhancedBiddingAgent(agent_name=f"Aggressive_Agent_{i}", strategy_profile="aggressive") for i in range(1, 3)
    ] + [
        EnhancedBiddingAgent(agent_name=f"Balanced_Agent_{i}", strategy_profile="balanced") for i in range(1, 3)
    ] + [
        EnhancedBiddingAgent(agent_name=f"Conservative_Agent_{1}", strategy_profile="conservative")
    ]
    simulation = EnhancedBiddingSimulation(agents=agents, rounds=20, initial_threshold=100)

    # Run Simulation
    simulation.run_simulation()
    simulation.summarize_results()
    
    # Log Performance Summary
    print("\n🏆 Simulation Performance Summary:")
    for agent in agents:
        metrics = agent.get_performance_metrics()
        print(f"Agent {agent.agent_name}: Win Rate={metrics['win_rate']:.2%}, Avg Bid={metrics['avg_bid']:.2f}")

    # Visualization (if enabled)
    if args.visualize and hasattr(simulation, "visualize_bidding_trends"):
        simulation.visualize_bidding_trends()

if __name__ == "__main__":
    main()
