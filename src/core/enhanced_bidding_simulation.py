import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from src.market.market_threshold import dynamic_market_threshold
import logging

logger = logging.getLogger("BiddingSystem")

class EnhancedBiddingSimulation:
    """
    Enhanced bidding simulation that works with EnhancedBiddingAgent.
    Provides competitor bids information and supports historical data integration.
    """
    
    def __init__(self, agents, rounds=20, initial_threshold=100, data_file="data/bid_history.csv", 
                 cost_price=None):
        self.agents = agents
        self.rounds = rounds
        self.current_threshold = initial_threshold
        self.original_threshold = initial_threshold  # Keep original threshold
        self.data_file = data_file
        self.bid_history = []
        self.cost_price = cost_price
        
        # Ensure data directory exists
        if not os.path.exists("data"):
            os.makedirs("data")
        
        # Clear previous data
        if os.path.exists(data_file):
            os.remove(data_file)
    
    def run_simulation(self) -> pd.DataFrame:
        """Run the enhanced bidding simulation."""
        
        print(f"🚀 Starting Enhanced Bidding Simulation")
        print(f"📊 Agents: {len(self.agents)}")
        print(f"🔄 Rounds: {self.rounds}")
        print(f"💰 Initial Threshold: {self.current_threshold}")
        print(f"🎯 Original Threshold (User Setting): {self.original_threshold}")
        
        all_data = []
        
        for round_num in range(1, self.rounds + 1):
            print(f"\n🛒 Round {round_num} - Market Threshold: {self.current_threshold:.2f}")
            
            # Track last winning bid for state information
            last_winning_bid = None
            if round_num > 1 and hasattr(self, 'last_round_winning_bid'):
                last_winning_bid = self.last_round_winning_bid
            
            # First pass: collect all bids
            bids = {}  # Initialize bids dictionary
            competitor_bids = []  # Initialize competitor bids list
            
            for agent in self.agents:
                # For the first round, use empty competitor bids
                if round_num == 1:
                    current_competitor_bids = []
                else:
                    # Use bids from previous round as competitor information
                    current_competitor_bids = list(bids.values())
                
                bid = agent.generate_bid(
                    market_threshold=self.current_threshold,
                    rounds_remaining=self.rounds - round_num,
                    competitor_bids=current_competitor_bids,
                    cost_price=self.cost_price,
                    last_winning_bid=last_winning_bid
                )
                
                bids[agent.agent_name] = bid
                competitor_bids.append(bid)
            
            # Validate bids against market threshold - be more strict
            valid_bids = {}
            for agent_name, bid in bids.items():
                if bid <= self.current_threshold:
                    valid_bids[agent_name] = bid
                else:
                    # Cap the bid at 95% of threshold if it exceeds
                    capped_bid = self.current_threshold * 0.95
                    valid_bids[agent_name] = capped_bid
                    print(f"⚠️ {agent_name} bid {bid:.2f} exceeded threshold {self.current_threshold}. Capped at {capped_bid:.2f}")
            
            # Determine winning bid (lowest bid wins in reverse auction)
            winning_bid = min(valid_bids.values()) if valid_bids else None
            winning_agent = min(valid_bids, key=valid_bids.get) if valid_bids else None
            
            # Update bids to use valid bids only
            bids = valid_bids
            
            # Update agent rewards with proper RL structure
            for agent in self.agents:
                won = bids[agent.agent_name] == winning_bid
                agent.update_reward(
                    reward=10 if won else -1,
                    won=won,
                    winning_bid=winning_bid
                )
                
                # Update opponent information for adaptive learning
                opponent_bids = [bid for name, bid in bids.items() if name != agent.agent_name]
                agent.update_opponent_info(opponent_bids, winning_bid)
            
            # Store last winning bid for next round
            self.last_round_winning_bid = winning_bid
            
            # Update market threshold dynamically but respect original threshold
            suggested_threshold = dynamic_market_threshold(
                self.current_threshold,
                list(bids.values())
            )
            
            # Ensure threshold doesn't exceed original user setting
            self.current_threshold = min(suggested_threshold, self.original_threshold)
            
            # Store data for this round
            for agent_name, bid in bids.items():
                data_point = {
                    "Round": round_num,
                    "Agent": agent_name,
                    "Bid": bid,
                    "Winning_Bid": (bid == winning_bid),
                    "Market_Threshold": self.current_threshold,
                    "Rounds_Remaining": self.rounds - round_num
                }
                all_data.append(data_point)
            
            print(f"📌 Bids: {bids}")
            print(f"🏆 Winning Bid: {winning_bid:.2f} by {winning_agent}")
            print(f"💰 Market Threshold: {self.current_threshold:.2f}")
            print(f"📊 Bid Range: {min(bids.values()):.2f} - {max(bids.values()):.2f}")
            print(f"🤖 Agents: {[agent.agent_name for agent in self.agents]}")
            if max(bids.values()) > self.current_threshold:
                print(f"⚠️ Warning: Some bids exceeded market threshold!")
            
            # Store bid history for next round
            self.bid_history.append(bids)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(all_data)
        df.to_csv(self.data_file, index=False)
        
        # Print performance summary
        self.print_performance_summary()
        
        # Print hybrid AI-RL analysis for each agent
        print("\n" + "=" * 60)
        print("🤖 HYBRID AI-RL PERFORMANCE ANALYSIS")
        print("=" * 60)
        for agent in self.agents:
            agent.print_hybrid_analysis()
        
        return df
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents."""
        
        if not self.bid_history:
            return {}
        
        summary = {}
        for agent in self.agents:
            agent_name = agent.agent_name
            agent_bids = [round_bids.get(agent_name, 0) for round_bids in self.bid_history]
            wins = sum(1 for round_bids in self.bid_history if round_bids.get(agent_name) == min(round_bids.values()))
            
            summary[agent_name] = {
                "total_bids": len(agent_bids),
                "wins": wins,
                "win_rate": wins / len(agent_bids) if agent_bids else 0,
                "avg_bid": np.mean(agent_bids) if agent_bids else 0,
                "min_bid": min(agent_bids) if agent_bids else 0,
                "max_bid": max(agent_bids) if agent_bids else 0,
                "strategy_profile": getattr(agent, 'strategy_profile', 'unknown')
            }
        
        return summary
    
    def print_performance_summary(self):
        """Print a formatted performance summary."""
        
        summary = self.get_agent_performance_summary()
        
        print("\n" + "="*60)
        print("🏆 SIMULATION PERFORMANCE SUMMARY")
        print("="*60)
        
        for agent_name, stats in summary.items():
            print(f"\n🤖 {agent_name} ({stats['strategy_profile']})")
            print(f"   📊 Win Rate: {stats['win_rate']:.1%}")
            print(f"   🏆 Wins: {stats['wins']}/{stats['total_bids']}")
            print(f"   💰 Average Bid: {stats['avg_bid']:.2f}")
            print(f"   📈 Bid Range: {stats['min_bid']:.2f} - {stats['max_bid']:.2f}")
        
        print("\n" + "="*60)
