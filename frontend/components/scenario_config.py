import streamlit as st
import pandas as pd
import os
from typing import Dict, Any, Optional

class ScenarioConfigurator:
    """Handles user input for auction scenario configuration."""
    
    def __init__(self):
        self.config = {}
    
    def render_configuration_panel(self) -> Dict[str, Any]:
        """Renders the scenario configuration panel in the sidebar."""
        
        st.sidebar.header("Auction Scenario Configuration")
        
        # Market Parameters Section
        st.sidebar.subheader("Market Parameters")
        
        # Number of bidders
        num_bidders = st.sidebar.number_input(
            "Number of Competing Bidders:",
            min_value=2,
            max_value=20,
            value=5,
            help="How many AI agents will compete in the auction"
        )
        
        # Market threshold
        market_threshold = st.sidebar.number_input(
            "Maximum Allowable Bid (Market Threshold):",
            min_value=50,
            max_value=1000,
            value=500,
            step=10,
            help="The ceiling bid that agents cannot exceed"
        )
        
        # NEW: Cost Price Input
        cost_price = st.sidebar.number_input(
            "Cost Price ($):",
            min_value=0.0,
            max_value=float(market_threshold),
            value=float(market_threshold * 0.3),
            step=10.0,
            help="Your production/service cost - bids below this will be unprofitable"
        )
        
        # Validate cost price
        if cost_price >= market_threshold:
            st.sidebar.error("Cost price cannot exceed market threshold!")
            return None
        
        # Auction rounds
        auction_rounds = st.sidebar.number_input(
            "Number of Auction Rounds:",
            min_value=10,
            max_value=100,
            value=50,
            help="How many rounds to simulate"
        )
        
        # Historical Data Section
        st.sidebar.subheader("Historical Data")
        
        # Option to upload historical data
        use_historical_data = st.sidebar.checkbox(
            "Use Historical Bidding Data",
            value=False,
            help="Upload past auction results to inform agent strategies"
        )
        
        historical_data = None
        if use_historical_data:
            uploaded_file = st.sidebar.file_uploader(
                "Upload Historical Auction Data (CSV):",
                type=['csv'],
                help="CSV file with columns: Round, Agent, Bid, Winning_Bid"
            )
            
            if uploaded_file is not None:
                try:
                    historical_data = pd.read_csv(uploaded_file)
                    st.sidebar.success(f"Loaded {len(historical_data)} historical records")
                    
                    # Show preview
                    if st.sidebar.checkbox("Preview Historical Data"):
                        st.sidebar.dataframe(historical_data.head())
                        
                except Exception as e:
                    st.sidebar.error(f"Error loading file: {e}")
        
        # Agent Strategy Profiles Section
        st.sidebar.subheader("Agent Strategy Profiles")
        
        # Strategy distribution
        aggressive_agents = st.sidebar.slider(
            "Aggressive Agents (Low Bids):",
            min_value=0,
            max_value=num_bidders,
            value=2,
            help="Agents that tend to bid very low"
        )
        
        conservative_agents = st.sidebar.slider(
            "Conservative Agents (High Bids):",
            min_value=0,
            max_value=num_bidders,
            value=1,
            help="Agents that start with higher bids"
        )
        
        # Calculate balanced agents
        balanced_agents = num_bidders - aggressive_agents - conservative_agents
        
        if balanced_agents < 0:
            st.sidebar.warning("Total agents exceed number of bidders!")
            balanced_agents = 0
        
        st.sidebar.info(f"Agent Distribution: {aggressive_agents} Aggressive, {balanced_agents} Balanced, {conservative_agents} Conservative")
        
        # Advanced Settings
        with st.sidebar.expander("Advanced Settings"):
            learning_rate = st.slider(
                "Agent Learning Rate:",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                help="How quickly agents learn from experience"
            )
            
            exploration_rate = st.slider(
                "Exploration Rate:",
                min_value=0.01,
                max_value=0.5,
                value=0.2,
                step=0.01,
                help="How often agents try random strategies"
            )
            
            ai_strategy_weight = st.slider(
                "AI Strategy Influence:",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="How much GPT-4 influences agent decisions"
            )
        
        # Validation
        if st.sidebar.button("Validate Configuration", type="primary"):
            if self._validate_configuration(num_bidders, market_threshold, cost_price, auction_rounds):
                st.sidebar.success("Configuration is valid!")
            else:
                st.sidebar.error("Configuration has issues!")
        
        # Return configuration
        return {
            "num_bidders": num_bidders,
            "market_threshold": market_threshold,
            "cost_price": cost_price,
            "auction_rounds": auction_rounds,
            "historical_data": historical_data,
            "agent_profiles": {
                "aggressive": aggressive_agents,
                "balanced": balanced_agents,
                "conservative": conservative_agents
            },
            "advanced_settings": {
                "learning_rate": learning_rate,
                "exploration_rate": exploration_rate,
                "ai_strategy_weight": ai_strategy_weight
            }
        }
    
    def _validate_configuration(self, num_bidders: int, market_threshold: float, 
                               cost_price: float, auction_rounds: int) -> bool:
        """Validates the enhanced user configuration."""
        
        if num_bidders < 2:
            st.error("Need at least 2 bidders for a competitive auction")
            return False
            
        if market_threshold <= 0:
            st.error("Market threshold must be positive")
            return False
            
        if cost_price >= market_threshold:
            st.error("Cost price cannot exceed market threshold")
            return False
            
        if auction_rounds < 10:
            st.error("Need at least 10 rounds for meaningful learning")
            return False
            
        return True
    
    def save_scenario(self, config: Dict[str, Any], scenario_name: str):
        """Saves the current scenario configuration."""
        
        scenarios_dir = "scenarios"
        if not os.path.exists(scenarios_dir):
            os.makedirs(scenarios_dir)
        
        scenario_file = os.path.join(scenarios_dir, f"{scenario_name}.json")
        
        # Convert pandas DataFrame to dict for JSON serialization
        if config.get("historical_data") is not None:
            config["historical_data"] = config["historical_data"].to_dict()
        
        import json
        with open(scenario_file, "w") as f:
            json.dump(config, f, indent=4)
        
        st.success(f"Scenario '{scenario_name}' saved successfully!")
    
    def load_scenario(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Loads a saved scenario configuration."""
        
        scenario_file = os.path.join("scenarios", f"{scenario_name}.json")
        
        if not os.path.exists(scenario_file):
            st.error(f"Scenario '{scenario_name}' not found!")
            return None
        
        import json
        with open(scenario_file, "r") as f:
            config = json.load(f)
        
        # Convert dict back to pandas DataFrame
        if config.get("historical_data") is not None:
            config["historical_data"] = pd.DataFrame(config["historical_data"])
        
        st.success(f"Scenario '{scenario_name}' loaded successfully!")
        return config
