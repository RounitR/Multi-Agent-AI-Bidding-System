import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.components.scenario_config import ScenarioConfigurator
from frontend.components.enhanced_visualization import EnhancedVisualization
from src.agents.enhanced_bidding_agent import EnhancedBiddingAgent
from src.core.recommendation_engine import OptimalBidRecommendationEngine, BidRecommendation
from src.core.enhanced_bidding_simulation import EnhancedBiddingSimulation
from src.utils.config import Config

class EnhancedBiddingPlatform:
    """
    Enhanced Multi-Agent AI Bidding Platform for Contractor Procurement.
    Implements all phases of the roadmap: User Input, Data-Driven Agents, 
    Optimal Recommendations, and Enhanced Visualizations.
    """
    
    def __init__(self):
        self.config = Config.load_config()
        self.scenario_config = ScenarioConfigurator()
        self.visualization = EnhancedVisualization()
        self.recommendation_engine = OptimalBidRecommendationEngine()
        
        # Initialize session state
        if 'simulation_data' not in st.session_state:
            st.session_state.simulation_data = None
        if 'recommendation' not in st.session_state:
            st.session_state.recommendation = None
        if 'scenario_config' not in st.session_state:
            st.session_state.scenario_config = None
    
    def run(self):
        """Main application runner."""
        
                # Page configuration
        st.set_page_config(
            page_title="Multi-Agent AI Bidding Simulator",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .recommendation-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
                # Header
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Multi-Agent AI Bidding Simulator</h1>
            <p>Advanced simulation platform for competitive bidding scenarios across any domain</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        self._render_sidebar()
        
        # Main content
        self._render_main_content()
    
    def _render_sidebar(self):
        """Render the sidebar with scenario configuration."""
        
        st.sidebar.title("🎯 Scenario Configuration")
        
        # Scenario management
        scenario_tab1, scenario_tab2 = st.sidebar.tabs(["Configure", "Manage"])
        
        with scenario_tab1:
            # Get scenario configuration
            scenario_config = self.scenario_config.render_configuration_panel()
            st.session_state.scenario_config = scenario_config
            
            # Run simulation button
            if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
                with st.spinner("Running AI simulation..."):
                    self._run_simulation(scenario_config)
        
        with scenario_tab2:
            # Scenario saving/loading
            st.subheader("💾 Save/Load Scenarios")
            
            # Save current scenario
            scenario_name = st.text_input("Scenario Name:")
            if st.button("💾 Save Scenario") and scenario_name and st.session_state.scenario_config:
                self.scenario_config.save_scenario(st.session_state.scenario_config, scenario_name)
            
            # Load saved scenario
            scenarios_dir = "scenarios"
            if os.path.exists(scenarios_dir):
                saved_scenarios = [f.replace('.json', '') for f in os.listdir(scenarios_dir) if f.endswith('.json')]
                if saved_scenarios:
                    selected_scenario = st.selectbox("Load Saved Scenario:", saved_scenarios)
                    if st.button("📂 Load Scenario"):
                        loaded_config = self.scenario_config.load_scenario(selected_scenario)
                        if loaded_config:
                            st.session_state.scenario_config = loaded_config
                            st.success(f"✅ Loaded scenario: {selected_scenario}")
        
        # AI Assistant
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 AI Assistant")
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize loading state
        if 'ai_loading' not in st.session_state:
            st.session_state.ai_loading = False
        
        # Initialize current response
        if 'current_response' not in st.session_state:
            st.session_state.current_response = None
        
        # Initialize input key for clearing
        if 'input_key' not in st.session_state:
            st.session_state.input_key = 0
        
        # Quick action buttons for common queries
        st.sidebar.markdown("**💡 Quick Questions:**")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📊 Strategy", help="Ask about bidding strategies", key="strategy_btn"):
                st.session_state.ai_loading = True
                response = self._get_enhanced_ai_response("What's the best bidding strategy?")
                st.session_state.current_response = response
                st.session_state.chat_history.append({"user": "What's the best bidding strategy?", "assistant": response})
                st.session_state.ai_loading = False
                st.session_state.input_key += 1
        
        with col2:
            if st.button("🎯 Recommendations", help="Ask about AI recommendations", key="recommendations_btn"):
                st.session_state.ai_loading = True
                response = self._get_enhanced_ai_response("How do AI recommendations work?")
                st.session_state.current_response = response
                st.session_state.chat_history.append({"user": "How do AI recommendations work?", "assistant": response})
                st.session_state.ai_loading = False
                st.session_state.input_key += 1
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            if st.button("📈 Analysis", help="Ask about result analysis", key="analysis_btn"):
                st.session_state.ai_loading = True
                response = self._get_enhanced_ai_response("How do I analyze the results?")
                st.session_state.current_response = response
                st.session_state.chat_history.append({"user": "How do I analyze the results?", "assistant": response})
                st.session_state.ai_loading = False
                st.session_state.input_key += 1
        
        with col4:
            if st.button("⚙️ Setup", help="Ask about configuration", key="setup_btn"):
                st.session_state.ai_loading = True
                response = self._get_enhanced_ai_response("How should I configure my scenario?")
                st.session_state.current_response = response
                st.session_state.chat_history.append({"user": "How should I configure my scenario?", "assistant": response})
                st.session_state.ai_loading = False
                st.session_state.input_key += 1
        
        # Show loading indicator
        if st.session_state.ai_loading:
            st.sidebar.info("🤖 AI is thinking... Please wait.")
        
        # Collapsible chat history
        if st.session_state.chat_history:
            with st.sidebar.expander("📚 Chat History", expanded=False):
                for i, message in enumerate(st.session_state.chat_history[:-1]):  # Show all except current
                    st.markdown(f"**👤 You:** {message['user']}")
                    st.markdown(f"**🤖 AI:** {message['assistant']}")
                    if i < len(st.session_state.chat_history[:-1]) - 1:  # Don't add separator after last message
                        st.markdown("---")
        
        # Chat interface
        st.sidebar.markdown("**💬 Chat with AI:**")
        
        # User input with form to prevent infinite loops
        with st.sidebar.form(key="chat_form"):
            user_query = st.text_input("Ask me anything about bidding:", key=f"ai_chat_input_{st.session_state.input_key}")
            submit_button = st.form_submit_button("Ask", disabled=st.session_state.ai_loading)
            
            if submit_button and user_query and not st.session_state.ai_loading:
                # Set loading state
                st.session_state.ai_loading = True
                
                # Get AI response using enhanced method
                ai_response = self._get_enhanced_ai_response(user_query)
                
                # Store current response and add to history
                st.session_state.current_response = ai_response
                st.session_state.chat_history.append({"user": user_query, "assistant": ai_response})
                
                # Clear loading state and increment input key to clear input
                st.session_state.ai_loading = False
                st.session_state.input_key += 1
                
                # Force rerun to clear input and show response
                st.rerun()
        
        # Show current response below input box
        if st.session_state.current_response and not st.session_state.ai_loading:
            st.sidebar.markdown("**🤖 AI Response:**")
            st.sidebar.markdown(st.session_state.current_response)
            st.sidebar.markdown("---")
        
        # Proactive insights based on current state
        if st.session_state.simulation_data is not None:
            st.sidebar.markdown("**💡 Proactive Insights:**")
            insights = self._get_proactive_insights()
            for insight in insights:
                st.sidebar.info(insight)
    
    def _render_main_content(self):
        """Render the main content area."""
        
        # Check if simulation has been run
        if st.session_state.simulation_data is None:
            self._render_welcome_screen()
        else:
            self._render_simulation_results()
    
    def _render_welcome_screen(self):
        """Render welcome screen when no simulation has been run."""
        
        st.markdown("""
        ## 🎯 Welcome to Multi-Agent AI Bidding Simulator
        
        This platform helps you simulate and analyze competitive bidding scenarios using advanced AI agents.
        
        ### 🚀 How it works:
        1. **Configure your scenario** in the sidebar
        2. **Upload historical data** (optional) for better insights
        3. **Run the simulation** with AI agents
        4. **Get optimal bid recommendations** based on market analysis
        5. **Explore detailed analytics** and insights
        
        ### 🎯 Key Features:
        - **Multi-Agent AI Simulation**: Realistic competitor behavior
        - **Historical Data Integration**: Learn from past auctions
        - **Optimal Bid Recommendations**: AI-powered strategy suggestions
        - **Comprehensive Analytics**: Detailed market insights
        - **Explainable AI**: Understand why recommendations are made
        
        ### 📊 What you'll get:
        - Optimal bid value with confidence score
        - Win probability analysis
        - Market dynamics insights
        - Competitor behavior analysis
        - Risk assessment
        - Alternative bid options
        
        ---
        
        **Ready to start?** Configure your scenario in the sidebar and run your first simulation!
        """)
        
        # Quick start example
        with st.expander("📖 Quick Start Example"):
            st.markdown("""
            **Example Scenario:**
            - **Number of Bidders**: 5
            - **Market Threshold**: $500,000
            - **Auction Rounds**: 50
            - **Agent Strategy**: 2 Aggressive, 2 Balanced, 1 Conservative
            
            This will simulate a competitive construction project bidding scenario with realistic AI agents.
            """)
    
    def _render_simulation_results(self):
        """Render simulation results and recommendations."""
        
        # Executive summary
        st.markdown("## 📊 Simulation Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Rounds",
                len(st.session_state.simulation_data['Round'].unique()),
                delta="Simulation Complete"
            )
        
        with col2:
            st.metric(
                "Number of Agents",
                len(st.session_state.simulation_data['Agent'].unique()),
                delta="Competitors"
            )
        
        with col3:
            avg_bid = st.session_state.simulation_data['Bid'].mean()
            st.metric(
                "Average Bid",
                f"${avg_bid:,.0f}",
                delta=f"Market Average"
            )
        
        with col4:
            if st.session_state.recommendation:
                optimal_bid = st.session_state.recommendation.recommended_bid
                st.metric(
                    "AI Recommended Bid",
                    f"${optimal_bid:,.0f}",
                    delta=f"{st.session_state.recommendation.win_probability:.1%} win probability"
                )
        
        # Recommendation highlight
        if st.session_state.recommendation:
            self._render_recommendation_highlight()
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Bidding Trends", 
            "🏆 Performance Analysis", 
            "📊 Market Dynamics",
            "🎯 Recommendation Details",
            "📋 Executive Summary"
        ])
        
        with tab1:
            self._render_bidding_trends()
        
        with tab2:
            self._render_performance_analysis()
        
        with tab3:
            self._render_market_dynamics()
        
        with tab4:
            self._render_recommendation_details()
        
        with tab5:
            self._render_executive_summary()
    
    def _render_recommendation_highlight(self):
        """Render the main recommendation highlight."""
        
        rec = st.session_state.recommendation
        
        st.markdown(f"""
        <div class="recommendation-box">
            <h3>🎯 AI Recommendation</h3>
            <h2>${rec.recommended_bid:,.2f}</h2>
            <p><strong>Win Probability:</strong> {rec.win_probability:.1%} | 
               <strong>Confidence:</strong> {rec.confidence_score:.1%} | 
               <strong>Risk Level:</strong> {rec.risk_level}</p>
            <p><em>{rec.reasoning}</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_bidding_trends(self):
        """Render bidding trends with insights."""
        
        self.visualization.plot_bidding_trends_with_insights(
            st.session_state.simulation_data,
            st.session_state.recommendation.__dict__ if st.session_state.recommendation else None
        )
        
        # Win probability heatmap
        self.visualization.plot_win_probability_heatmap(
            st.session_state.simulation_data,
            st.session_state.recommendation.__dict__ if st.session_state.recommendation else None
        )
    
    def _render_performance_analysis(self):
        """Render agent performance analysis."""
        
        self.visualization.plot_agent_performance_dashboard(st.session_state.simulation_data)
    
    def _render_market_dynamics(self):
        """Render market dynamics analysis."""
        
        self.visualization.plot_market_dynamics_analysis(st.session_state.simulation_data)
        
        # Historical comparison if available
        if st.session_state.scenario_config and st.session_state.scenario_config.get('historical_data') is not None:
            self.visualization.plot_historical_comparison(
                st.session_state.simulation_data,
                st.session_state.scenario_config['historical_data']
            )
    
    def _render_recommendation_details(self):
        """Render detailed recommendation analysis."""
        
        if st.session_state.recommendation:
            self.visualization.plot_recommendation_analysis(st.session_state.recommendation.__dict__)
        else:
            st.warning("No recommendation available. Please run a simulation first.")
    
    def _render_executive_summary(self):
        """Render executive summary."""
        
        summary = self.visualization.create_executive_summary(
            st.session_state.simulation_data,
            st.session_state.recommendation.__dict__ if st.session_state.recommendation else None,
            st.session_state.scenario_config if hasattr(st.session_state, 'scenario_config') else None
        )
        
        st.markdown(summary)
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download Simulation Data"):
                csv = st.session_state.simulation_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="bidding_simulation_results.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📄 Download Executive Summary"):
                # Create a simple text summary
                summary_text = f"""
                AI-Powered Procurement Simulation Report
                =========================================
                
                {summary}
                
                Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.download_button(
                    label="Download Report",
                    data=summary_text,
                    file_name="procurement_simulation_report.txt",
                    mime="text/plain"
                )
    
    def _run_simulation(self, scenario_config: Dict[str, Any]):
        """Run the bidding simulation with the given configuration."""
        
        try:
            # Create enhanced agents
            agents = []
            agent_profiles = scenario_config['agent_profiles']
            advanced_settings = scenario_config['advanced_settings']
            
            # Create agents based on strategy profiles
            agent_names = []
            
            # Aggressive agents
            for i in range(agent_profiles['aggressive']):
                agent_name = f"Aggressive_Agent_{i+1}"
                agent = EnhancedBiddingAgent(
                    agent_name=agent_name,
                    strategy_profile="aggressive",
                    historical_data=scenario_config.get('historical_data'),
                    learning_rate=advanced_settings['learning_rate'],
                    exploration_rate=advanced_settings['exploration_rate'],
                    ai_strategy_weight=advanced_settings['ai_strategy_weight']
                )
                agents.append(agent)
                agent_names.append(agent_name)
            
            # Balanced agents
            for i in range(agent_profiles['balanced']):
                agent_name = f"Balanced_Agent_{i+1}"
                agent = EnhancedBiddingAgent(
                    agent_name=agent_name,
                    strategy_profile="balanced",
                    historical_data=scenario_config.get('historical_data'),
                    learning_rate=advanced_settings['learning_rate'],
                    exploration_rate=advanced_settings['exploration_rate'],
                    ai_strategy_weight=advanced_settings['ai_strategy_weight']
                )
                agents.append(agent)
                agent_names.append(agent_name)
            
            # Conservative agents
            for i in range(agent_profiles['conservative']):
                agent_name = f"Conservative_Agent_{i+1}"
                agent = EnhancedBiddingAgent(
                    agent_name=agent_name,
                    strategy_profile="conservative",
                    historical_data=scenario_config.get('historical_data'),
                    learning_rate=advanced_settings['learning_rate'],
                    exploration_rate=advanced_settings['exploration_rate'],
                    ai_strategy_weight=advanced_settings['ai_strategy_weight']
                )
                agents.append(agent)
                agent_names.append(agent_name)
            
            # Create enhanced simulation
            simulation = EnhancedBiddingSimulation(
                agents=agents,
                rounds=scenario_config['auction_rounds'],
                initial_threshold=scenario_config['market_threshold'],
                cost_price=scenario_config.get('cost_price')
            )
            
            # Run simulation
            simulation_data = simulation.run_simulation()
            
            # Store results
            st.session_state.simulation_data = simulation_data
            st.session_state.scenario_config = scenario_config
            
            # Generate recommendation with cost constraints
            recommendation = self.recommendation_engine.analyze_simulation_results(
                simulation_data,
                scenario_config['market_threshold'],
                "User_Agent",  # Assuming user is represented by one agent
                scenario_config.get('cost_price')
            )
            
            st.session_state.recommendation = recommendation
            
            st.success("✅ Simulation completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")
            st.exception(e)
    
    def _get_ai_response(self, query: str) -> str:
        """Get AI response for user queries."""
        
        try:
            # Enhanced AI response system with more specific answers
            query_lower = query.lower()
            
            # Strategy-related questions
            if any(word in query_lower for word in ["strategy", "strategic", "approach", "tactic"]):
                return "**Bidding Strategy**: The AI simulation uses a hybrid approach combining Deep Q-Learning, historical data analysis, and GPT-4 insights. Aggressive agents bid lower for higher win probability, while conservative agents prioritize profit margins. The optimal strategy balances winning probability with profitability based on market conditions and competitor behavior."
            
            # Risk-related questions
            elif any(word in query_lower for word in ["risk", "danger", "safe", "conservative"]):
                return "**Risk Assessment**: Lower bids increase win probability but reduce profit margins. The AI analyzes market volatility, competitor aggression levels, and bid consistency to assess risk. Conservative strategies minimize risk but may miss opportunities, while aggressive strategies maximize win probability but reduce profitability."
            
            # Competition-related questions
            elif any(word in query_lower for word in ["competitor", "competition", "opponent", "rival"]):
                return "**Competitor Analysis**: The system tracks each agent's bidding patterns, learning rates, and strategy types (aggressive/balanced/conservative). You can see win rates, average bids, and volatility for each competitor. This helps you anticipate their moves and position your bid strategically."
            
            # Historical data questions
            elif any(word in query_lower for word in ["historical", "history", "past", "data", "trend"]):
                return "**Historical Data Integration**: Upload past auction results to improve AI learning. The system analyzes historical winning bids, bid distributions, and market trends to inform agent strategies. This makes simulations more realistic and recommendations more accurate."
            
            # Recommendation questions
            elif any(word in query_lower for word in ["recommend", "suggestion", "optimal", "best", "advice"]):
                return "**AI Recommendations**: The system provides optimal bid suggestions with confidence scores and win probabilities. Recommendations are based on simulation outcomes, market analysis, and competitor behavior. They balance winning probability with profit optimization for your specific scenario."
            
            # Market threshold questions
            elif any(word in query_lower for word in ["threshold", "market", "price", "limit"]):
                return "**Market Threshold**: This is the maximum allowable bid in the auction. The system dynamically adjusts thresholds based on market conditions while respecting your original setting. Higher thresholds allow more flexibility but may increase competition."
            
            # Win probability questions
            elif any(word in query_lower for word in ["win", "probability", "chance", "success"]):
                return "**Win Probability**: Calculated based on your bid relative to competitor bids and market conditions. Higher bids reduce win probability but increase profit margins. The AI recommends the optimal balance for your scenario."
            
            # Agent behavior questions
            elif any(word in query_lower for word in ["agent", "behavior", "learning", "adapt"]):
                return "**Agent Behavior**: AI agents learn and adapt over time using Deep Q-Learning. They have different strategy profiles (aggressive/balanced/conservative) and learn from experience. You can see their performance metrics and how they improve throughout the simulation."
            
            # General help
            elif any(word in query_lower for word in ["help", "how", "what", "explain", "tell"]):
                return "**How to Use This Platform**: 1) Configure your scenario (bidders, threshold, rounds), 2) Run the simulation to see AI agents compete, 3) Get optimal bid recommendations with confidence scores, 4) Analyze results with interactive charts and insights. The system helps you make data-driven bidding decisions!"
            
            # Default response
            else:
                return "I can help you with bidding strategies, risk assessment, competitor analysis, historical data integration, AI recommendations, market thresholds, win probabilities, and agent behavior. Try asking about specific aspects like 'What's the best bidding strategy?' or 'How does risk assessment work?'"
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def _get_enhanced_ai_response(self, query: str) -> str:
        """Get enhanced AI response with context awareness and detailed insights."""
        
        try:
            query_lower = query.lower()
            
            # Context-aware responses based on simulation state
            has_simulation_data = st.session_state.simulation_data is not None
            has_recommendation = hasattr(st.session_state, 'recommendation') and st.session_state.recommendation is not None
            
            # Strategy-related questions with enhanced context
            if any(word in query_lower for word in ["strategy", "strategic", "approach", "tactic", "best"]):
                if has_simulation_data:
                    return self._get_strategy_response_with_context()
                else:
                    return """**🎯 Bidding Strategy Guide:**

**Hybrid AI-RL Approach**: Our system combines Deep Q-Learning with GPT-4 insights for optimal decision-making.

**Strategy Profiles:**
- **Aggressive**: Lower bids, higher win probability, lower profit margins
- **Balanced**: Optimal balance between winning and profitability  
- **Conservative**: Higher bids, lower win probability, higher profit margins

**Key Factors:**
- Market threshold and competition intensity
- Historical data patterns
- Real-time competitor behavior analysis
- Dynamic confidence-based AI-RL weighting

**Pro Tip**: Start with balanced agents and adjust based on your risk tolerance!"""
            
            # Risk assessment with simulation data
            elif any(word in query_lower for word in ["risk", "danger", "safe", "conservative", "volatile"]):
                if has_simulation_data:
                    return self._get_risk_analysis_with_context()
                else:
                    return """**⚠️ Risk Assessment Framework:**

**Risk Factors:**
- **Market Volatility**: High bid variance indicates unstable market
- **Competition Intensity**: More competitors = higher risk
- **Bid Positioning**: Lower bids = higher win probability but lower profit
- **Historical Patterns**: Past performance predicts future risk

**Risk Mitigation:**
- Use conservative agents for stable markets
- Increase AI weight for volatile conditions
- Monitor competitor behavior patterns
- Balance win probability with profit margins

**Risk Levels:**
- **Low**: Stable market, predictable competitors
- **Medium**: Moderate volatility, balanced competition
- **High**: High volatility, aggressive competitors"""
            
            # Competition analysis with real data
            elif any(word in query_lower for word in ["competitor", "competition", "opponent", "rival", "other"]):
                if has_simulation_data:
                    return self._get_competitor_analysis_with_context()
                else:
                    return """**🏆 Competitor Analysis:**

**What We Track:**
- Individual agent performance metrics
- Bidding patterns and strategies
- Learning rates and adaptation speed
- Win rates and bid volatility

**Strategy Types:**
- **Aggressive**: Low bids, high win rates, low profits
- **Balanced**: Moderate bids, balanced performance
- **Conservative**: High bids, low win rates, high profits

**Key Insights:**
- Monitor competitor win rates
- Analyze bid distribution patterns
- Track learning and adaptation
- Identify strategy shifts over time"""
            
            # Results analysis with specific data
            elif any(word in query_lower for word in ["analyze", "analysis", "results", "interpret", "understand"]):
                if has_simulation_data:
                    return self._get_results_analysis_with_context()
                else:
                    return """**📊 Results Analysis Guide:**

**Key Metrics to Monitor:**
- **Win Rates**: Success frequency for each agent
- **Average Bids**: Market positioning and strategy effectiveness
- **Bid Ranges**: Volatility and risk assessment
- **Learning Curves**: Agent improvement over time

**Visualizations:**
- Bidding trends over rounds
- Performance comparison charts
- Market dynamics analysis
- Risk assessment indicators

**Actionable Insights:**
- Identify optimal bid ranges
- Understand market dynamics
- Assess competition intensity
- Optimize strategy selection"""
            
            # Configuration guidance
            elif any(word in query_lower for word in ["configure", "setup", "settings", "parameters", "scenario", "example", "test"]):
                return """**⚙️ Scenario Configuration Guide:**

**Example Scenario for Testing:**
- **Number of Bidders**: 5 (2 Aggressive, 2 Balanced, 1 Conservative)
- **Market Threshold**: $500,000
- **Cost Price**: $300,000
- **Auction Rounds**: 30
- **AI Strategy Weight**: 0.7

**Essential Parameters:**
- **Number of Bidders**: 3-10 recommended for realistic competition
- **Market Threshold**: Set based on project budget and market rates
- **Auction Rounds**: 20-50 rounds for meaningful learning
- **Cost Price**: Your minimum viable bid (profit floor)

**Agent Distribution:**
- **Aggressive**: 30-40% for competitive scenarios
- **Balanced**: 40-50% for standard scenarios  
- **Conservative**: 20-30% for high-value scenarios

**Advanced Settings:**
- **Learning Rate**: 0.1-0.3 for stable learning
- **Exploration Rate**: 0.2-0.4 for strategy discovery
- **AI Strategy Weight**: 0.5-0.8 for balanced AI-RL approach

**Pro Tips:**
- Start with balanced distribution
- Use historical data when available
- Adjust based on market conditions
- Monitor performance metrics"""
            
            # AI recommendations explanation
            elif any(word in query_lower for word in ["recommend", "suggestion", "optimal", "best", "advice"]):
                if has_recommendation:
                    return self._get_recommendation_explanation_with_context()
                else:
                    return """**🎯 AI Recommendation System:**

**How It Works:**
- Analyzes simulation outcomes and market dynamics
- Considers competitor behavior and historical patterns
- Balances win probability with profit optimization
- Provides confidence scores and risk assessments

**Recommendation Components:**
- **Optimal Bid**: Best balance of winning and profit
- **Win Probability**: Likelihood of winning with this bid
- **Confidence Score**: Reliability of the recommendation
- **Risk Level**: Associated risk assessment
- **Alternative Options**: Backup strategies

**Factors Considered:**
- Market threshold and competition
- Historical winning patterns
- Competitor aggression levels
- Cost constraints and profit margins
- Market volatility and stability"""
            
            # General help and guidance
            elif any(word in query_lower for word in ["help", "how", "what", "explain", "tell", "guide", "example", "test"]):
                return """**🚀 How to Use This Platform:**

**Step-by-Step Guide:**
1. **Configure Scenario**: Set bidders, threshold, rounds in sidebar
2. **Upload Data**: Add historical data for better insights (optional)
3. **Run Simulation**: Watch AI agents compete and learn
4. **Analyze Results**: Review performance metrics and trends
5. **Get Recommendations**: Receive optimal bid suggestions
6. **Optimize Strategy**: Adjust parameters based on insights

**Example Scenario for Testing:**
- **Number of Bidders**: 5 (2 Aggressive, 2 Balanced, 1 Conservative)
- **Market Threshold**: $500,000
- **Cost Price**: $300,000
- **Auction Rounds**: 30
- **AI Strategy Weight**: 0.7

**Key Features:**
- **Multi-Agent AI Simulation**: Realistic competitor behavior
- **Hybrid AI-RL System**: Best of both worlds
- **Historical Data Integration**: Learn from past auctions
- **Comprehensive Analytics**: Detailed market insights
- **Interactive Visualizations**: Understand trends and patterns

**Pro Tips:**
- Start with the example scenario above
- Use the AI Assistant for guidance on any step
- Monitor agent performance for insights
- Experiment with different configurations"""
            
            # Default response with suggestions
            else:
                return """**🤖 I'm here to help with your bidding simulation!**

**I can assist with:**
- 📊 **Strategy guidance** and best practices
- 🎯 **AI recommendations** and optimization
- 📈 **Results analysis** and interpretation
- ⚙️ **Configuration** and setup advice
- ⚠️ **Risk assessment** and mitigation
- 🏆 **Competitor analysis** and insights

**Try asking:**
- "What's the best bidding strategy?"
- "How do I analyze the results?"
- "What should I configure for my scenario?"
- "How do AI recommendations work?"
- "What's the risk level in my simulation?"

**Or use the quick action buttons above for instant answers!**"""
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _get_strategy_response_with_context(self) -> str:
        """Get strategy response with current simulation context."""
        
        data = st.session_state.simulation_data
        avg_bid = data['Bid'].mean()
        win_rate = (data['Winning_Bid'] == 1).mean()
        volatility = data['Bid'].std() / avg_bid
        
        return f"""**🎯 Strategy Analysis (Based on Your Simulation):**

**Current Market Conditions:**
- Average Bid: ${avg_bid:,.0f}
- Win Rate: {win_rate:.1%}
- Market Volatility: {volatility:.1%}

**Strategy Insights:**
- **Market Position**: {'Competitive' if avg_bid < data['Market_Threshold'].iloc[0] * 0.7 else 'Conservative'}
- **Risk Level**: {'High' if volatility > 0.3 else 'Medium' if volatility > 0.15 else 'Low'}
- **Competition Intensity**: {'High' if len(data['Agent'].unique()) > 5 else 'Moderate'}

**Recommendations:**
- {'Consider more aggressive bidding for higher win rates' if win_rate < 0.3 else 'Maintain current strategy for balanced performance'}
- {'Increase AI weight for better strategic guidance' if volatility > 0.3 else 'Current AI-RL balance appears optimal'}
- {'Monitor competitor learning patterns' if len(data['Round'].unique()) > 10 else 'Run more rounds for better insights'}"""
    
    def _get_risk_analysis_with_context(self) -> str:
        """Get risk analysis with current simulation context."""
        
        data = st.session_state.simulation_data
        volatility = data['Bid'].std() / data['Bid'].mean()
        competition = len(data['Agent'].unique())
        win_rate = (data['Winning_Bid'] == 1).mean()
        
        risk_level = "High" if volatility > 0.3 or competition > 7 else "Medium" if volatility > 0.15 or competition > 5 else "Low"
        
        return f"""**⚠️ Risk Assessment (Based on Your Simulation):**

**Risk Factors:**
- **Market Volatility**: {volatility:.1%} ({'High' if volatility > 0.3 else 'Medium' if volatility > 0.15 else 'Low'})
- **Competition Level**: {competition} agents ({'High' if competition > 7 else 'Medium' if competition > 5 else 'Low'})
- **Win Rate**: {win_rate:.1%} ({'Low risk' if win_rate > 0.4 else 'Medium risk' if win_rate > 0.2 else 'High risk'})

**Overall Risk Level: {risk_level}**

**Risk Mitigation Strategies:**
- {'Use conservative agents to reduce volatility' if volatility > 0.3 else 'Current agent mix is appropriate'}
- {'Consider reducing competition or increasing market threshold' if competition > 7 else 'Competition level is manageable'}
- {'Focus on strategic positioning over aggressive bidding' if win_rate < 0.2 else 'Current strategy is effective'}"""
    
    def _get_competitor_analysis_with_context(self) -> str:
        """Get competitor analysis with current simulation context."""
        
        data = st.session_state.simulation_data
        
        # Analyze each agent's performance
        agent_analysis = []
        for agent in data['Agent'].unique():
            agent_data = data[data['Agent'] == agent]
            win_rate = (agent_data['Winning_Bid'] == 1).mean()
            avg_bid = agent_data['Bid'].mean()
            volatility = agent_data['Bid'].std() / avg_bid
            
            strategy = "Aggressive" if avg_bid < data['Bid'].mean() * 0.9 else "Conservative" if avg_bid > data['Bid'].mean() * 1.1 else "Balanced"
            
            agent_analysis.append(f"**{agent}**: {strategy} strategy, {win_rate:.1%} win rate, ${avg_bid:,.0f} avg bid")
        
        return f"""**🏆 Competitor Analysis (Based on Your Simulation):**

**Market Overview:**
- Total Competitors: {len(data['Agent'].unique())}
- Average Win Rate: {(data['Winning_Bid'] == 1).mean():.1%}
- Market Average Bid: ${data['Bid'].mean():,.0f}

**Individual Agent Performance:**
{chr(10).join(agent_analysis)}

**Key Insights:**
- {'High competition with diverse strategies' if len(data['Agent'].unique()) > 5 else 'Moderate competition with focused strategies'}
- {'Volatile market with unpredictable outcomes' if data['Bid'].std() / data['Bid'].mean() > 0.3 else 'Stable market with predictable patterns'}
- {'Learning agents adapting strategies' if len(data['Round'].unique()) > 10 else 'Early stage with limited learning data'}"""
    
    def _get_results_analysis_with_context(self) -> str:
        """Get results analysis with current simulation context."""
        
        data = st.session_state.simulation_data
        
        return f"""**📊 Results Analysis (Your Simulation):**

**Simulation Overview:**
- Total Rounds: {len(data['Round'].unique())}
- Total Agents: {len(data['Agent'].unique())}
- Total Bids: {len(data)}

**Performance Metrics:**
- Average Bid: ${data['Bid'].mean():,.0f}
- Bid Range: ${data['Bid'].min():,.0f} - ${data['Bid'].max():,.0f}
- Market Volatility: {data['Bid'].std() / data['Bid'].mean():.1%}
- Overall Win Rate: {(data['Winning_Bid'] == 1).mean():.1%}

**Key Insights:**
- {'High market efficiency with competitive pricing' if data['Bid'].std() / data['Bid'].mean() < 0.2 else 'Market volatility indicates strategic opportunities'}
- {'Balanced competition with fair win distribution' if 0.1 < (data['Winning_Bid'] == 1).mean() < 0.3 else 'Unbalanced competition requiring strategy adjustment'}
- {'Sufficient data for reliable analysis' if len(data) > 50 else 'Consider running more rounds for better insights'}

**Next Steps:**
- Review agent performance individually
- Analyze bidding trends over time
- Consider parameter adjustments
- Use recommendations for optimization"""
    
    def _get_recommendation_explanation_with_context(self) -> str:
        """Get recommendation explanation with current recommendation context."""
        
        rec = st.session_state.recommendation
        
        return f"""**🎯 AI Recommendation Analysis (Your Results):**

**Optimal Bid: ${rec.recommended_bid:,.2f}**
- Win Probability: {rec.win_probability:.1%}
- Confidence Score: {rec.confidence_score:.1%}
- Risk Level: {rec.risk_level}

**Recommendation Rationale:**
- Based on {len(st.session_state.simulation_data)} simulation data points
- Considers {len(st.session_state.simulation_data['Agent'].unique())} competitor strategies
- Balances win probability ({rec.win_probability:.1%}) with profit optimization
- Accounts for market volatility and competition intensity

**Alternative Options:**
{chr(10).join([f"- ${bid:,.2f} (Win Prob: {prob:.1%})" for bid, prob in rec.alternative_bids[:3]])}

**Strategic Advice:**
- {'Consider more aggressive bidding for higher win rates' if rec.win_probability < 0.4 else 'Current recommendation balances winning and profit well'}
- {'Monitor market conditions for volatility changes' if rec.risk_level == 'High' else 'Market conditions appear stable'}
- {'Use this as baseline and adjust based on risk tolerance' if rec.confidence_score > 0.7 else 'Consider additional analysis for higher confidence'}"""
    
    def _get_proactive_insights(self) -> List[str]:
        """Get proactive insights based on current simulation state."""
        
        insights = []
        data = st.session_state.simulation_data
        
        if data is not None:
            # Analyze win rate
            win_rate = (data['Winning_Bid'] == 1).mean()
            if win_rate < 0.2:
                insights.append("⚠️ Low win rate detected. Consider more aggressive bidding strategy.")
            elif win_rate > 0.4:
                insights.append("🎉 High win rate! Your strategy is performing well.")
            
            # Analyze volatility
            volatility = data['Bid'].std() / data['Bid'].mean()
            if volatility > 0.3:
                insights.append("📈 High market volatility detected. Consider increasing AI weight for strategic guidance.")
            
            # Analyze competition
            if len(data['Agent'].unique()) > 7:
                insights.append("🏆 High competition detected. Focus on strategic positioning over aggressive bidding.")
            
            # Suggest more rounds
            if len(data['Round'].unique()) < 20:
                insights.append("🔄 Consider running more rounds for better learning and insights.")
        
        return insights

def main():
    """Main entry point for the enhanced application."""
    
    app = EnhancedBiddingPlatform()
    app.run()

if __name__ == "__main__":
    main()
