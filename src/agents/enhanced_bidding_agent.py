import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from typing import Dict, List, Optional, Tuple
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class EnhancedBiddingAgent:
    """
    Enhanced bidding agent with historical data integration and configurable strategy profiles.
    Implements Phase 2 of the roadmap: Data-Driven Competitor Agents & Learning Integration.
    """
    
    def __init__(self, 
                 agent_name: str,
                 strategy_profile: str = "balanced",
                 historical_data: Optional[pd.DataFrame] = None,
                 learning_rate: float = 0.1,
                 exploration_rate: float = 0.2,
                 ai_strategy_weight: float = 0.7):
        
        self.agent_name = agent_name
        self.strategy_profile = strategy_profile
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.ai_strategy_weight = ai_strategy_weight
        
        # Initialize strategy parameters based on profile
        self._initialize_strategy_profile()
        
        # Initialize DQN network with discretized action space
        self.state_size = 6  # [threshold, rounds_left, last_winning_bid, rolling_avg_bid, bid_std, cost/threshold]
        self.action_size = 31  # Discrete bid actions (K=31 bins)
        self.q_network = self._build_q_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Discretized action space
        self.action_bins = None  # Will be set when cost_price is known
        self.current_cost_price = None
        self.current_market_threshold = None
        
        # Historical data integration
        self.historical_data = historical_data
        self.historical_stats = None
        self.agent_stats = None
        if historical_data is not None:
            self._preprocess_historical_data()
        
        # Experience replay buffer
        self.memory = []
        
        # Performance tracking for hybrid AI-RL system
        self.ai_suggestions = []
        self.rl_bids = []
        self.ai_confidence_history = []
        self.rl_confidence_history = []
        self.decision_history = []
        self.win_history = []
        self.memory_size = 1000
        
        # Performance tracking
        self.wins = 0
        self.total_bids = 0
        self.bid_history = []
        
        # Opponent tracking
        self.opponent_bid_history = []
        self.winning_bid_history = []
        self.moving_avg_winning_bid = None
        self.opponent_distribution = None
        
        # OpenAI client
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def set_action_space(self, cost_price: float, market_threshold: float):
        """Set the discretized action space based on cost and threshold."""
        self.current_cost_price = cost_price
        self.current_market_threshold = market_threshold
        
        # Create 31 bins from cost_price to market_threshold
        self.action_bins = np.linspace(cost_price, market_threshold, self.action_size)
        print(f"🎯 {self.agent_name} action space: {len(self.action_bins)} bins from {cost_price:.2f} to {market_threshold:.2f}")
    
    def update_opponent_info(self, opponent_bids: List[float], winning_bid: float):
        """Update opponent information for adaptive learning."""
        
        # Store opponent bids
        self.opponent_bid_history.extend(opponent_bids)
        
        # Store winning bid
        self.winning_bid_history.append(winning_bid)
        
        # Calculate moving average of winning bids (last 10 rounds)
        if len(self.winning_bid_history) > 0:
            recent_wins = self.winning_bid_history[-10:]  # Last 10 winning bids
            self.moving_avg_winning_bid = np.mean(recent_wins)
        
        # Calculate opponent bid distribution
        if len(self.opponent_bid_history) > 0:
            recent_opponent_bids = self.opponent_bid_history[-20:]  # Last 20 opponent bids
            self.opponent_distribution = {
                'mean': np.mean(recent_opponent_bids),
                'std': np.std(recent_opponent_bids),
                'min': np.min(recent_opponent_bids),
                'max': np.max(recent_opponent_bids),
                'percentile_25': np.percentile(recent_opponent_bids, 25),
                'percentile_75': np.percentile(recent_opponent_bids, 75)
            }
    
    def _initialize_strategy_profile(self):
        """Initialize agent behavior based on strategy profile."""
        
        if self.strategy_profile == "aggressive":
            self.base_bid_multiplier = 0.6  # Start with competitive bids (60% of threshold)
            self.learning_aggression = 1.5   # Learn quickly from losses
            self.exploration_bias = 0.3      # Prefer lower bids during exploration
            
        elif self.strategy_profile == "conservative":
            self.base_bid_multiplier = 0.9   # Start with higher bids (90% of threshold)
            self.learning_aggression = 0.7   # Learn more slowly
            self.exploration_bias = 0.8      # Prefer higher bids during exploration
            
        else:  # balanced
            self.base_bid_multiplier = 0.75   # Start with medium bids (75% of threshold)
            self.learning_aggression = 1.0   # Normal learning rate
            self.exploration_bias = 0.5      # No bias in exploration
    
    def _preprocess_historical_data(self):
        """Preprocess historical data to inform agent strategy."""
        
        if self.historical_data is None:
            return
        
        # Calculate historical statistics
        self.historical_stats = {
            'avg_winning_bid': self.historical_data['Bid'].mean(),
            'std_winning_bid': self.historical_data['Bid'].std(),
            'min_winning_bid': self.historical_data['Bid'].min(),
            'max_winning_bid': self.historical_data['Bid'].max(),
            'winning_bid_percentile_25': self.historical_data['Bid'].quantile(0.25),
            'winning_bid_percentile_75': self.historical_data['Bid'].quantile(0.75)
        }
        
        # Calculate agent-specific statistics if available
        if 'Agent' in self.historical_data.columns:
            agent_data = self.historical_data[self.historical_data['Agent'] == self.agent_name]
            if not agent_data.empty:
                self.agent_stats = {
                    'avg_bid': agent_data['Bid'].mean(),
                    'win_rate': (agent_data['Winning_Bid'] == 1).mean(),
                    'best_bid': agent_data['Bid'].min()
                }
            else:
                self.agent_stats = None
        else:
            self.agent_stats = None
        
        print(f"📊 {self.agent_name} loaded historical data: {len(self.historical_data)} records")
    
    def _build_q_network(self) -> nn.Module:
        """Build the Deep Q-Network."""
        
        class QNetwork(nn.Module):
            def __init__(self, state_size, action_size):
                super(QNetwork, self).__init__()
                self.fc1 = nn.Linear(state_size, 64)
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, action_size)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)
        
        return QNetwork(self.state_size, self.action_size)
    
    def _get_state_vector(self, market_threshold: float, rounds_remaining: int, 
                         competitor_bids: List[float], last_winning_bid: float = None) -> torch.Tensor:
        """Convert environment state to neural network input."""
        
        avg_bid = np.mean(competitor_bids) if competitor_bids else market_threshold * 0.5
        std_bid = np.std(competitor_bids) if len(competitor_bids) > 1 else market_threshold * 0.1
        last_winning_bid = last_winning_bid if last_winning_bid else market_threshold * 0.5
        
        # Normalize state values
        state = [
            market_threshold / 1000.0,  # Normalize threshold
            rounds_remaining / 100.0,   # Normalize rounds
            last_winning_bid / market_threshold,  # Normalize last winning bid
            avg_bid / market_threshold, # Normalize average bid
            std_bid / market_threshold,  # Normalize bid standard deviation
            self.current_cost_price / market_threshold if self.current_cost_price else 0.5  # Normalize cost ratio
        ]
        
        return torch.FloatTensor(state)
    
    def _get_ai_strategy_suggestion(self, market_threshold: float, 
                                  rounds_remaining: int, 
                                  competitor_bids: List[float]) -> float:
        """Get AI strategy suggestion from GPT-4 with enhanced market analysis."""
        
        if self.openai_client is None:
            return None
        
        try:
            # Enhanced market analysis
            market_analysis = self._analyze_market_context(market_threshold, competitor_bids, rounds_remaining)
            
            # Build comprehensive context
            context = self._build_ai_context(market_analysis)
            
            prompt = f"""
            You are an AI expert in competitive bidding strategies for {self.strategy_profile} agents.
            
            Current market situation:
            - Market threshold: {market_threshold}
            - Rounds remaining: {rounds_remaining}
            - Number of competitors: {len(competitor_bids)}
            - Agent strategy profile: {self.strategy_profile}
            
            Market Analysis:
            {market_analysis}
            
            {context}
            
            Based on this comprehensive analysis, suggest an optimal bid value (as a number only) that maximizes the chance of winning while maintaining profitability for a {self.strategy_profile} agent.
            Consider the current market conditions, competitor behavior, and strategic positioning.
            Respond with ONLY a number representing the suggested bid.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI bidding strategy expert. Respond with ONLY a number representing the suggested bid. Do not include any text, explanations, or other characters - just the number."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1  # Lower temperature for more consistent numerical output
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to extract number from response
            try:
                import re
                number_match = re.search(r'\d+\.?\d*', content)
                if number_match:
                    suggestion = float(number_match.group())
                else:
                    suggestion = float(content)
                
                # Validate suggestion is within reasonable bounds
                if suggestion < market_threshold * 0.1 or suggestion > market_threshold * 1.1:
                    print(f"⚠️ AI suggestion {suggestion} outside reasonable bounds, using fallback")
                    return None
                
                print(f"🤖 {self.agent_name} AI Suggestion: {suggestion:.2f}")
                return suggestion
            except (ValueError, TypeError):
                print(f"⚠️ Could not parse AI suggestion: '{content}'. Using fallback strategy.")
                return None
            
        except Exception as e:
            print(f"⚠️ AI strategy error for {self.agent_name}: {e}")
            return None
    
    def _analyze_market_context(self, market_threshold: float, competitor_bids: List[float], rounds_remaining: int) -> str:
        """Analyze current market context for AI decision making."""
        
        if len(competitor_bids) == 0:
            return "No competitor data available yet."
        
        avg_bid = np.mean(competitor_bids)
        bid_std = np.std(competitor_bids)
        min_bid = min(competitor_bids)
        max_bid = max(competitor_bids)
        
        # Market volatility assessment
        volatility = bid_std / avg_bid if avg_bid > 0 else 0.0
        if volatility > 0.3:
            volatility_level = "High"
        elif volatility > 0.15:
            volatility_level = "Medium"
        else:
            volatility_level = "Low"
        
        # Market positioning analysis
        threshold_utilization = avg_bid / market_threshold
        if threshold_utilization < 0.4:
            market_position = "Underserved (low competition)"
        elif threshold_utilization < 0.7:
            market_position = "Balanced"
        else:
            market_position = "Overserved (high competition)"
        
        # Strategic timing analysis
        if rounds_remaining > 10:
            timing = "Early game - can be more aggressive"
        elif rounds_remaining > 5:
            timing = "Mid game - balanced approach"
        else:
            timing = "Late game - conservative approach"
        
        analysis = f"""
        Market Statistics:
        - Average competitor bid: {avg_bid:.2f}
        - Bid range: {min_bid:.2f} - {max_bid:.2f}
        - Bid standard deviation: {bid_std:.2f}
        - Market volatility: {volatility_level} ({volatility:.2%})
        - Market threshold utilization: {threshold_utilization:.2%}
        - Market position: {market_position}
        - Strategic timing: {timing}
        """
        
        return analysis
    
    def _build_ai_context(self, market_analysis: str) -> str:
        """Build comprehensive context for AI decision making."""
        
        context = ""
        
        # Historical performance context
        if hasattr(self, 'historical_stats') and self.historical_stats is not None:
            try:
                context += f"""
                Historical Market Context:
                - Average winning bid: {self.historical_stats['avg_winning_bid']:.2f}
                - Winning bid range: {self.historical_stats['min_winning_bid']:.2f} - {self.historical_stats['max_winning_bid']:.2f}
                - 25th percentile: {self.historical_stats['winning_bid_percentile_25']:.2f}
                - 75th percentile: {self.historical_stats['winning_bid_percentile_75']:.2f}
                """
            except (KeyError, TypeError):
                pass
        
        # Agent performance context
        if hasattr(self, 'agent_stats') and self.agent_stats is not None:
            try:
                context += f"""
                Agent Performance History:
                - Average bid: {self.agent_stats['avg_bid']:.2f}
                - Win rate: {self.agent_stats['win_rate']:.2%}
                - Best performing bid: {self.agent_stats['best_bid']:.2f}
                """
            except (KeyError, TypeError):
                pass
        
        # Strategy profile guidance
        if self.strategy_profile == "aggressive":
            context += """
            Strategy Profile: Aggressive
            - Focus on winning probability over profit margin
            - Can bid closer to cost price
            - Target lower bid ranges for higher win rates
            """
        elif self.strategy_profile == "conservative":
            context += """
            Strategy Profile: Conservative
            - Focus on profit margin over winning probability
            - Maintain higher bid values
            - Target upper bid ranges for better profitability
            """
        else:  # balanced
            context += """
            Strategy Profile: Balanced
            - Balance winning probability with profit margin
            - Target middle bid ranges
            - Adapt based on market conditions
            """
        
        return context
    
    def generate_bid(self, market_threshold: float, rounds_remaining: int, 
                    competitor_bids: List[float], cost_price: float = None, 
                    last_winning_bid: float = None) -> float:
        """Generate a bid using hybrid AI-RL approach with discretized action space."""
        
        # Set action space if not already set
        if self.action_bins is None and cost_price:
            self.set_action_space(cost_price, market_threshold)
        
        # Get state vector
        state = self._get_state_vector(market_threshold, rounds_remaining, competitor_bids, last_winning_bid)
        
        # Get AI strategy suggestion
        ai_suggestion = self._get_ai_strategy_suggestion(market_threshold, rounds_remaining, competitor_bids)
        
        # Epsilon-greedy action selection with profile-specific bias
        if random.random() < self.exploration_rate:
            # Exploration: Random action with profile-specific bias
            action_idx = self._exploration_with_bias()
        else:
            # Exploitation: Best action from Q-network
            action_idx = self._exploitation_strategy(state)
        
        # Get RL bid from action index
        if self.action_bins is not None:
            rl_bid = self.action_bins[action_idx]
        else:
            # Fallback to old strategy if action space not set
            rl_bid = self._fallback_strategy(market_threshold, competitor_bids)
        
        # Hybrid decision making: Combine AI suggestion with RL bid
        if ai_suggestion is not None:
            bid = self._hybrid_decision_making(ai_suggestion, rl_bid, market_threshold, competitor_bids)
        else:
            bid = rl_bid
        
        # Apply cost constraint if available
        if cost_price:
            bid = max(bid, cost_price)
        
        # Ensure bid is within valid range
        bid = min(bid, market_threshold)
        
        # Record bid and action for training
        self.bid_history.append(bid)
        self.total_bids += 1
        
        # Track decision type for performance analysis
        if ai_suggestion is not None:
            self.decision_type = "hybrid"
            self.ai_suggestions.append(ai_suggestion)
            self.rl_bids.append(rl_bid)
        else:
            self.decision_type = "rl_only"
            self.rl_bids.append(rl_bid)
        
        return bid
    
    def get_hybrid_performance_metrics(self) -> Dict[str, any]:
        """Get performance metrics for the hybrid AI-RL system."""
        
        metrics = {
            'total_decisions': self.total_bids,
            'hybrid_decisions': len(self.ai_suggestions),
            'rl_only_decisions': self.total_bids - len(self.ai_suggestions),
            'ai_utilization_rate': len(self.ai_suggestions) / self.total_bids if self.total_bids > 0 else 0,
            'avg_ai_confidence': 0.0,
            'avg_rl_confidence': 0.0,
            'hybrid_effectiveness': 0.0
        }
        
        # Calculate average confidences
        if hasattr(self, 'ai_confidence_history') and self.ai_confidence_history:
            metrics['avg_ai_confidence'] = np.mean(self.ai_confidence_history)
        
        if hasattr(self, 'rl_confidence_history') and self.rl_confidence_history:
            metrics['avg_rl_confidence'] = np.mean(self.rl_confidence_history)
        
        # Calculate hybrid effectiveness (win rate when AI was used)
        if len(self.ai_suggestions) > 0 and hasattr(self, 'win_history'):
            hybrid_wins = 0
            hybrid_decisions = 0
            
            for i, decision_type in enumerate(self.decision_history):
                if decision_type == "hybrid" and i < len(self.win_history):
                    hybrid_decisions += 1
                    if self.win_history[i]:
                        hybrid_wins += 1
            
            if hybrid_decisions > 0:
                metrics['hybrid_effectiveness'] = hybrid_wins / hybrid_decisions
        
        return metrics
    
    def _track_decision_performance(self, ai_confidence: float, rl_confidence: float, decision_type: str):
        """Track decision performance for analysis."""
        
        # Initialize tracking lists if they don't exist
        if not hasattr(self, 'ai_confidence_history'):
            self.ai_confidence_history = []
        if not hasattr(self, 'rl_confidence_history'):
            self.rl_confidence_history = []
        if not hasattr(self, 'decision_history'):
            self.decision_history = []
        
        # Record confidences and decision type
        self.ai_confidence_history.append(ai_confidence)
        self.rl_confidence_history.append(rl_confidence)
        self.decision_history.append(decision_type)
    
    def print_hybrid_analysis(self):
        """Print analysis of hybrid AI-RL performance."""
        
        metrics = self.get_hybrid_performance_metrics()
        
        print(f"\n🤖 {self.agent_name} Hybrid AI-RL Analysis:")
        print(f"   📊 Total Decisions: {metrics['total_decisions']}")
        print(f"   🧠 AI Utilization: {metrics['ai_utilization_rate']:.1%}")
        print(f"   🎯 Hybrid Effectiveness: {metrics['hybrid_effectiveness']:.1%}")
        print(f"   📈 Avg AI Confidence: {metrics['avg_ai_confidence']:.2f}")
        print(f"   🧮 Avg RL Confidence: {metrics['avg_rl_confidence']:.2f}")
        
        if len(self.ai_suggestions) > 0:
            print(f"   💡 AI Suggestions Range: {min(self.ai_suggestions):.2f} - {max(self.ai_suggestions):.2f}")
        if len(self.rl_bids) > 0:
            print(f"   🎲 RL Bids Range: {min(self.rl_bids):.2f} - {max(self.rl_bids):.2f}")
    
    def _exploration_strategy(self, market_threshold: float, competitor_bids: List[float]) -> float:
        """Generate bid during exploration phase."""
        
        # Use historical data if available
        if hasattr(self, 'historical_stats') and self.historical_stats is not None:
            try:
                # Sample from historical distribution
                historical_range = (
                    self.historical_stats['winning_bid_percentile_25'],
                    self.historical_stats['winning_bid_percentile_75']
                )
                base_bid = random.uniform(*historical_range)
            except (KeyError, TypeError):
                # Fallback to strategy profile bias if historical data is incomplete
                base_bid = market_threshold * self.base_bid_multiplier
        else:
            # Use strategy profile bias
            base_bid = market_threshold * self.base_bid_multiplier
        
        # Add exploration noise
        noise = random.uniform(-0.2, 0.2) * market_threshold
        bid = base_bid + noise
        
        return bid
    
    def _exploitation_strategy(self, state: torch.Tensor) -> int:
        """Get best action index from Q-network."""
        
        # Get Q-values from network
        with torch.no_grad():
            q_values = self.q_network(state)
        
        # Select action index with highest Q-value
        action_idx = q_values.argmax().item()
        
        return action_idx
    
    def _exploration_with_bias(self) -> int:
        """Exploration with profile-specific bias."""
        
        if self.action_bins is None:
            return random.randint(0, self.action_size - 1)
        
        # Profile-specific exploration bias
        if self.strategy_profile == "aggressive":
            # Prefer lower bid indices (lower bids)
            bias_range = (0, self.action_size // 2)
        elif self.strategy_profile == "conservative":
            # Prefer higher bid indices (higher bids)
            bias_range = (self.action_size // 2, self.action_size)
        else:  # balanced
            # No bias, explore all actions
            bias_range = (0, self.action_size)
        
        # Add some randomness to avoid getting stuck
        if random.random() < 0.3:  # 30% chance to explore outside bias range
            return random.randint(0, self.action_size - 1)
        else:
            return random.randint(bias_range[0], bias_range[1] - 1)
    
    def _fallback_strategy(self, market_threshold: float, competitor_bids: List[float]) -> float:
        """Fallback strategy when action space is not set."""
        
        # Use base bid multiplier
        base_bid = market_threshold * self.base_bid_multiplier
        
        # Add some randomness
        noise = random.uniform(-0.1, 0.1) * market_threshold
        bid = base_bid + noise
        
        return bid
    
    def _hybrid_decision_making(self, ai_suggestion: float, rl_bid: float, market_threshold: float, competitor_bids: List[float]) -> float:
        """
        Hybrid decision-making logic that combines AI suggestions with RL bids based on confidence levels.
        """
        
        # Assess confidence levels
        ai_confidence = self._assess_ai_confidence(market_threshold, competitor_bids)
        rl_confidence = self._assess_rl_confidence()
        
        # Adapt strategy based on market conditions
        self._adapt_strategy(market_threshold, competitor_bids)
        
        # Calculate effective weights
        effective_ai_weight = self.ai_strategy_weight * ai_confidence
        effective_rl_weight = (1 - self.ai_strategy_weight) * rl_confidence
        
        # Normalize weights
        total_weight = effective_ai_weight + effective_rl_weight
        if total_weight > 0:
            effective_ai_weight /= total_weight
            effective_rl_weight /= total_weight
        else:
            # Fallback to equal weights if both confidences are 0
            effective_ai_weight = 0.5
            effective_rl_weight = 0.5
        
        # Combine decisions
        if ai_suggestion is not None:
            hybrid_bid = (ai_suggestion * effective_ai_weight) + (rl_bid * effective_rl_weight)
            decision_type = "hybrid"
            print(f"🤖 {self.agent_name} Hybrid Decision: AI({ai_suggestion:.2f} × {effective_ai_weight:.2f}) + RL({rl_bid:.2f} × {effective_rl_weight:.2f}) = {hybrid_bid:.2f}")
        else:
            hybrid_bid = rl_bid
            decision_type = "rl_only"
            print(f"🤖 {self.agent_name} RL Only: {rl_bid:.2f} (AI suggestion unavailable)")
        
        # Track decision performance
        self._track_decision_performance(ai_confidence, rl_confidence, decision_type)
        
        return hybrid_bid
    
    def _assess_ai_confidence(self, market_threshold: float, competitor_bids: List[float]) -> float:
        """
        Assess confidence in AI suggestions based on market conditions.
        Returns confidence score between 0.0 and 1.0.
        """
        confidence = 0.5  # Base confidence
        
        # Market stability assessment
        if len(competitor_bids) > 0:
            bid_std = np.std(competitor_bids)
            bid_mean = np.mean(competitor_bids)
            stability = 1.0 - (bid_std / bid_mean) if bid_mean > 0 else 0.0
            confidence += stability * 0.2
        
        # Historical data quality
        if self.historical_stats is not None:
            confidence += 0.2
        
        # Market threshold utilization
        if len(competitor_bids) > 0:
            avg_bid = np.mean(competitor_bids)
            threshold_utilization = avg_bid / market_threshold
            if 0.3 < threshold_utilization < 0.8:  # Sweet spot
                confidence += 0.1
        
        # Competitor behavior predictability
        if len(competitor_bids) >= 3:
            # Check if competitors are following predictable patterns
            bid_trend = np.polyfit(range(len(competitor_bids)), competitor_bids, 1)[0]
            if abs(bid_trend) < np.mean(competitor_bids) * 0.1:  # Stable trend
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _assess_rl_confidence(self) -> float:
        """
        Assess confidence in RL decisions based on training history.
        Returns confidence score between 0.0 and 1.0.
        """
        confidence = 0.5  # Base confidence
        
        # Training experience
        if self.total_bids > 50:
            confidence += 0.2
        elif self.total_bids > 20:
            confidence += 0.1
        
        # Recent performance
        if len(self.bid_history) >= 5:
            recent_wins = sum(1 for i, bid in enumerate(self.bid_history[-5:]) 
                            if i < len(self.win_history) and self.win_history[i])
            recent_win_rate = recent_wins / 5
            confidence += recent_win_rate * 0.2
        
        # Q-network stability
        if hasattr(self, 'q_network') and self.q_network is not None:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _adapt_strategy(self, market_threshold: float, competitor_bids: List[float]):
        """
        Adapt strategy based on market conditions.
        """
        if len(competitor_bids) == 0:
            return
        
        # Calculate market volatility
        bid_std = np.std(competitor_bids)
        bid_mean = np.mean(competitor_bids)
        volatility = bid_std / bid_mean if bid_mean > 0 else 0.0
        
        # Adjust exploration rate based on volatility
        if volatility > 0.3:  # High volatility
            self.exploration_rate = min(self.exploration_rate * 1.2, 0.8)
        elif volatility < 0.1:  # Low volatility
            self.exploration_rate = max(self.exploration_rate * 0.8, 0.1)
        
        # Adjust AI strategy weight based on market conditions
        if volatility > 0.4:  # Very high volatility - rely more on AI for strategic guidance
            self.ai_strategy_weight = min(self.ai_strategy_weight * 1.3, 1.0)
        elif volatility < 0.05:  # Very low volatility - rely more on RL for optimization
            self.ai_strategy_weight = max(self.ai_strategy_weight * 0.7, 0.1)
    
    def update_reward(self, reward: float, state: torch.Tensor = None, action_idx: int = None, 
                     next_state: torch.Tensor = None, won: bool = False, winning_bid: float = None):
        """Update agent with reward and train Q-network."""
        
        # Track win history for performance analysis
        self.win_history.append(won)
        
        # Store experience in replay buffer
        if state is not None and action_idx is not None:
            self.memory.append((state, action_idx, reward, next_state))
            
            # Train Q-network if we have enough samples
            if len(self.memory) >= 10:
                self._train_q_network()
        
        # Update exploration rate (epsilon annealing)
        if self.total_bids > 0:
            self.exploration_rate = max(0.05, 0.4 - (self.total_bids * 0.007))
    
    def _train_q_network(self):
        """Train the Q-network using experience replay."""
        
        if len(self.memory) < 32:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, 32)
        
        # Prepare training data
        states = []
        actions = []
        rewards = []
        
        for experience in batch:
            if experience['state'] is not None:
                states.append(experience['state'])
                actions.append(experience['action'] if experience['action'] is not None else 0)
                rewards.append(experience['reward'])
        
        if not states:
            return
        
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        
        # Get current Q-values
        current_q_values = self.q_network(states)
        
        # Create target Q-values
        target_q_values = current_q_values.clone()
        
        # Update Q-values for the actions that were taken
        for i in range(len(actions)):
            if actions[i] < self.action_size:
                target_q_values[i][actions[i]] = rewards[i]
        
        # Train network
        self.optimizer.zero_grad()
        loss = nn.MSELoss()(current_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get agent performance metrics."""
        
        return {
            'win_rate': self.wins / max(self.total_bids, 1),
            'avg_bid': np.mean(self.bid_history) if self.bid_history else 0,
            'total_bids': self.total_bids,
            'wins': self.wins,
            'exploration_rate': self.exploration_rate
        }
    
    def get_strategy_summary(self) -> str:
        """Get a summary of the agent's strategy and performance."""
        
        metrics = self.get_performance_metrics()
        
        summary = f"""
        🤖 {self.agent_name} Strategy Summary:
        - Profile: {self.strategy_profile}
        - Win Rate: {metrics['win_rate']:.2%}
        - Average Bid: {metrics['avg_bid']:.2f}
        - Total Bids: {metrics['total_bids']}
        - Exploration Rate: {metrics['exploration_rate']:.3f}
        """
        
        if hasattr(self, 'historical_stats') and self.historical_stats is not None:
            try:
                summary += f"""
        - Historical Data: {len(self.historical_data)} records loaded
        - Historical Avg Winning Bid: {self.historical_stats['avg_winning_bid']:.2f}
        """
            except (KeyError, TypeError, AttributeError):
                summary += f"""
        - Historical Data: Not available or incomplete
        """
        
        return summary
