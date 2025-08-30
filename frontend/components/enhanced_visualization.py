import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

class EnhancedVisualization:
    """
    Enhanced visualization components for Phase 4 of the roadmap.
    Provides comprehensive analytics with explainable AI overlays.
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.theme = "plotly_white"
    
    def plot_bidding_trends_with_insights(self, df: pd.DataFrame, recommendation: Optional[Dict] = None):
        """Enhanced bidding trends with AI insights and recommendation overlay."""
        
        st.subheader("Bidding Trends with AI Insights")
        
        # Create interactive plot
        fig = go.Figure()
        
        # Plot each agent's bidding trend
        for agent in df['Agent'].unique():
            agent_data = df[df['Agent'] == agent]
            
            fig.add_trace(go.Scatter(
                x=agent_data['Round'],
                y=agent_data['Bid'],
                mode='lines+markers',
                name=agent,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{agent}</b><br>' +
                             'Round: %{x}<br>' +
                             'Bid: %{y:.2f}<br>' +
                             '<extra></extra>'
            ))
        
        # Add recommendation overlay if available
        if recommendation:
            optimal_bid = recommendation.get('recommended_bid', 0)
            fig.add_hline(
                y=optimal_bid,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Recommended Bid: {optimal_bid:.2f}",
                annotation_position="top right"
            )
        
        # Add market threshold line
        threshold = df['Bid'].max() * 1.1  # Approximate threshold
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Market Threshold: {threshold:.2f}",
            annotation_position="bottom right"
        )
        
        # Update layout
        fig.update_layout(
            title="Bidding Trends Over Rounds with AI Recommendations",
            xaxis_title="Round Number",
            yaxis_title="Bid Value",
            hovermode='x unified',
            template=self.theme,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights panel
        self._add_trend_insights(df, recommendation)
    
    def plot_win_probability_heatmap(self, df: pd.DataFrame, recommendation: Optional[Dict] = None):
        """Win probability heatmap showing optimal bidding zones."""
        
        st.subheader("Win Probability Heatmap")
        
        # Calculate win probabilities for different bid ranges
        bid_ranges = np.linspace(df['Bid'].min(), df['Bid'].max(), 20)
        win_probs = []
        
        for bid in bid_ranges:
            # Calculate how often this bid would win
            wins = (df['Bid'] > bid).sum()
            total = len(df)
            win_prob = wins / total if total > 0 else 0
            win_probs.append(win_prob)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[win_probs],
            x=bid_ranges,
            y=['Win Probability'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Win Probability")
        ))
        
        # Add recommendation marker
        if recommendation:
            optimal_bid = recommendation.get('recommended_bid', 0)
            fig.add_vline(
                x=optimal_bid,
                line_color="red",
                line_width=3,
                annotation_text=f"Optimal: {optimal_bid:.2f}"
            )
        
        fig.update_layout(
            title="Win Probability by Bid Value",
            xaxis_title="Bid Value",
            yaxis_title="",
            template=self.theme,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_agent_performance_dashboard(self, df: pd.DataFrame):
        """Comprehensive agent performance dashboard."""
        
        st.subheader("Agent Performance Dashboard")
        
        # Calculate performance metrics
        performance_data = []
        for agent in df['Agent'].unique():
            agent_data = df[df['Agent'] == agent]
            
            metrics = {
                'Agent': agent,
                'Win Rate': (agent_data['Winning_Bid'] == 1).mean(),
                'Avg Bid': agent_data['Bid'].mean(),
                'Bid Volatility': agent_data['Bid'].std(),
                'Total Bids': len(agent_data),
                'Best Bid': agent_data['Bid'].min(),
                'Worst Bid': agent_data['Bid'].max()
            }
            performance_data.append(metrics)
        
        perf_df = pd.DataFrame(performance_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win Rate', 'Average Bid', 'Bid Volatility', 'Performance Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Win Rate
        fig.add_trace(
            go.Bar(x=perf_df['Agent'], y=perf_df['Win Rate'], name='Win Rate'),
            row=1, col=1
        )
        
        # Average Bid
        fig.add_trace(
            go.Bar(x=perf_df['Agent'], y=perf_df['Avg Bid'], name='Avg Bid'),
            row=1, col=2
        )
        
        # Bid Volatility
        fig.add_trace(
            go.Bar(x=perf_df['Agent'], y=perf_df['Bid Volatility'], name='Volatility'),
            row=2, col=1
        )
        
        # Performance Score (Win Rate vs Avg Bid efficiency)
        performance_score = perf_df['Win Rate'] * (1 - perf_df['Avg Bid'] / perf_df['Avg Bid'].max())
        fig.add_trace(
            go.Scatter(x=perf_df['Agent'], y=performance_score, mode='markers+text',
                      text=[f'{score:.2f}' for score in performance_score],
                      textposition='top center', name='Performance Score'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, template=self.theme)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.subheader("Detailed Performance Metrics")
        st.dataframe(perf_df.round(3))
    
    def plot_market_dynamics_analysis(self, df: pd.DataFrame):
        """Advanced market dynamics analysis."""
        
        st.subheader("Market Dynamics Analysis")
        
        # Calculate market statistics over time
        market_stats = df.groupby('Round').agg({
            'Bid': ['mean', 'std', 'min', 'max'],
            'Winning_Bid': 'sum'
        }).reset_index()
        
        market_stats.columns = ['Round', 'Avg_Bid', 'Bid_Std', 'Min_Bid', 'Max_Bid', 'Wins']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bid Statistics Over Time', 'Bid Range Evolution', 
                          'Market Volatility', 'Competition Intensity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Bid Statistics
        fig.add_trace(
            go.Scatter(x=market_stats['Round'], y=market_stats['Avg_Bid'], 
                      name='Average Bid', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=market_stats['Round'], y=market_stats['Min_Bid'], 
                      name='Minimum Bid', line=dict(color='green')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=market_stats['Round'], y=market_stats['Max_Bid'], 
                      name='Maximum Bid', line=dict(color='red')),
            row=1, col=1
        )
        
        # Bid Range
        bid_range = market_stats['Max_Bid'] - market_stats['Min_Bid']
        fig.add_trace(
            go.Scatter(x=market_stats['Round'], y=bid_range, 
                      name='Bid Range', line=dict(color='purple')),
            row=1, col=2
        )
        
        # Market Volatility
        fig.add_trace(
            go.Scatter(x=market_stats['Round'], y=market_stats['Bid_Std'], 
                      name='Bid Standard Deviation', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Competition Intensity (number of unique bids per round)
        competition_intensity = df.groupby('Round')['Bid'].nunique()
        fig.add_trace(
            go.Scatter(x=competition_intensity.index, y=competition_intensity.values, 
                      name='Unique Bids', line=dict(color='brown')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, template=self.theme)
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_recommendation_analysis(self, recommendation: Dict):
        """Visualize bid recommendation analysis."""
        
        st.subheader("AI Bid Recommendation Analysis")
        
        # Create recommendation summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Recommended Bid",
                f"${recommendation['recommended_bid']:.2f}",
                delta=f"{recommendation['confidence_score']:.1%} confidence"
            )
        
        with col2:
            st.metric(
                "Win Probability",
                f"{recommendation['win_probability']:.1%}",
                delta=f"{recommendation['win_probability'] - 0.5:.1%} vs random"
            )
        
        with col3:
            confidence_level = "High" if recommendation['confidence_score'] > 0.7 else "Medium" if recommendation['confidence_score'] > 0.4 else "Low"
            st.metric(
                "Confidence Level",
                confidence_level,
                delta=f"{recommendation['confidence_score']:.1%}"
            )
        
        with col4:
            # Map risk levels to valid Streamlit delta colors
            risk_color_map = {"Low": "normal", "Medium": "normal", "High": "inverse"}
            st.metric(
                "Risk Level",
                recommendation['risk_level'],
                delta_color=risk_color_map.get(recommendation['risk_level'], "normal")
            )
        
        # Alternative bids comparison
        st.subheader("Alternative Bid Options")
        
        alternatives = recommendation.get('alternative_bids', [])
        if alternatives:
            alt_df = pd.DataFrame(alternatives, columns=['Bid', 'Win_Probability'])
            alt_df['Profit_Margin'] = 1 - (alt_df['Bid'] / alt_df['Bid'].max())
            alt_df['Score'] = alt_df['Win_Probability'] * alt_df['Profit_Margin']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=alt_df['Bid'],
                y=alt_df['Win_Probability'],
                mode='markers+text',
                text=alt_df.index + 1,
                textposition='top center',
                marker=dict(
                    size=alt_df['Score'] * 20 + 10,
                    color=alt_df['Score'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Score")
                ),
                name='Alternative Bids'
            ))
            
            # Highlight recommended bid
            if 'recommended_bid' in recommendation:
                fig.add_vline(
                    x=recommendation['recommended_bid'],
                    line_color="red",
                    line_width=3,
                    annotation_text="Recommended"
                )
            
            fig.update_layout(
                title="Alternative Bids: Win Probability vs Bid Value",
                xaxis_title="Bid Value",
                yaxis_title="Win Probability",
                template=self.theme
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # AI Explanation - Technical Details
        st.subheader("AI Explanation")
        
        # Create a more detailed technical explanation for recommendation details
        technical_explanation = self._generate_technical_explanation(recommendation, st.session_state.get('scenario_config'))
        st.info(technical_explanation)
    
    def _generate_technical_explanation(self, recommendation: Dict, scenario_config: Optional[Dict] = None) -> str:
        """Generate a detailed technical explanation for recommendation details with cost analysis."""
        
        recommended_bid = recommendation.get('recommended_bid', 0)
        win_probability = recommendation.get('win_probability', 0)
        confidence_score = recommendation.get('confidence_score', 0)
        risk_level = recommendation.get('risk_level', 'Unknown')
        
        # Get market analysis data if available
        market_analysis = recommendation.get('market_analysis', {})
        avg_bid = market_analysis.get('avg_bid', 0)
        winning_bid_avg = market_analysis.get('winning_bid_avg', 0)
        competition_intensity = market_analysis.get('competition_intensity', 0)
        
        # Get cost information if available
        cost_price = scenario_config.get('cost_price', None) if scenario_config else None
        
        explanation = f"""
        **Technical Analysis Summary:**
        
        The recommended bid of **${recommended_bid:.2f}** is calculated considering both market conditions and competitor behavior. 
        The winning bid average of **${winning_bid_avg:.2f}** suggests that {'lower' if recommended_bid < winning_bid_avg else 'higher'} bids are currently effective in the market.
        """
        
        # Add cost-based analysis if available
        if cost_price:
            profit_margin = ((recommended_bid - cost_price) / cost_price * 100) if cost_price > 0 else 0
            is_profitable = recommended_bid >= cost_price
            
            explanation += f"""
        
        **Cost-Based Analysis:**
        - Cost price: ${cost_price:.2f}
        - Recommended bid profit margin: {profit_margin:.1f}%
        - Profitability status: {'Profitable' if is_profitable else 'Below cost'}
        
        The recommendation {'ensures' if is_profitable else 'does not ensure'} cost coverage while maintaining competitive positioning.
        """
        
        explanation += f"""
        
        **Market Context:**
        - Average market bid: ${avg_bid:.2f}
        - Competition intensity: {competition_intensity} agents
        - Win probability: {win_probability:.1%}
        - Confidence level: {confidence_score:.1%}
        - Risk assessment: {risk_level}
        
        **Strategic Rationale:**
        By strategically placing our bid {'slightly below' if recommended_bid < winning_bid_avg else 'slightly above'} the winning average, 
        we aim to enhance our chances of success while maintaining profitable margins. 
        The {confidence_score:.1%} confidence level indicates {'strong' if confidence_score > 0.7 else 'moderate' if confidence_score > 0.4 else 'limited'} 
        statistical support for this recommendation.
        """
        
        return explanation.strip()
    
    def plot_historical_comparison(self, current_data: pd.DataFrame, historical_data: Optional[pd.DataFrame] = None):
        """Compare current simulation with historical data."""
        
        st.subheader("Historical Comparison Analysis")
        
        if historical_data is None:
            st.warning("No historical data available for comparison.")
            return
        
        # Create comparison metrics
        current_stats = {
            'avg_bid': current_data['Bid'].mean(),
            'winning_bid_avg': current_data[current_data['Winning_Bid'] == 1]['Bid'].mean(),
            'bid_volatility': current_data['Bid'].std(),
            'competition_level': len(current_data['Agent'].unique())
        }
        
        historical_stats = {
            'avg_bid': historical_data['Bid'].mean(),
            'winning_bid_avg': historical_data[historical_data['Winning_Bid'] == 1]['Bid'].mean(),
            'bid_volatility': historical_data['Bid'].std(),
            'competition_level': len(historical_data['Agent'].unique())
        }
        
        # Create comparison chart
        metrics = ['avg_bid', 'winning_bid_avg', 'bid_volatility']
        metric_names = ['Average Bid', 'Winning Bid Average', 'Bid Volatility']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current Simulation',
            x=metric_names,
            y=[current_stats[m] for m in metrics],
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            name='Historical Data',
            x=metric_names,
            y=[historical_stats[m] for m in metrics],
            marker_color='orange'
        ))
        
        fig.update_layout(
            title="Current vs Historical Performance",
            barmode='group',
            template=self.theme
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        st.subheader("Trend Analysis")
        
        # Compare bid distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.hist(current_data['Bid'], alpha=0.7, label='Current', bins=20)
        ax1.hist(historical_data['Bid'], alpha=0.7, label='Historical', bins=20)
        ax1.set_title('Bid Distribution Comparison')
        ax1.legend()
        
        ax2.boxplot([current_data['Bid'], historical_data['Bid']], 
                   labels=['Current', 'Historical'])
        ax2.set_title('Bid Range Comparison')
        
        st.pyplot(fig)
    
    def _add_trend_insights(self, df: pd.DataFrame, recommendation: Optional[Dict] = None):
        """Add AI-powered insights about bidding trends."""
        
        st.subheader("AI-Powered Trend Insights")
        
        # Calculate insights
        insights = []
        
        # Trend direction
        early_bids = df[df['Round'] <= df['Round'].max() // 2]['Bid'].mean()
        late_bids = df[df['Round'] > df['Round'].max() // 2]['Bid'].mean()
        
        if late_bids < early_bids:
            insights.append("**Bidding is becoming more aggressive** - Agents are learning to bid lower over time.")
        else:
            insights.append("**Bidding is becoming more conservative** - Agents are increasing their bids.")
        
        # Competition analysis
        unique_bids_per_round = df.groupby('Round')['Bid'].nunique().mean()
        if unique_bids_per_round > len(df['Agent'].unique()) * 0.8:
            insights.append("**High competition diversity** - Agents are using different strategies.")
        else:
            insights.append("**Low competition diversity** - Agents are converging on similar strategies.")
        
        # Volatility analysis
        bid_volatility = df.groupby('Agent')['Bid'].std().mean()
        if bid_volatility > df['Bid'].std() * 0.5:
            insights.append("**High agent volatility** - Some agents are experimenting with different strategies.")
        else:
            insights.append("**Low agent volatility** - Agents are maintaining consistent strategies.")
        
        # Recommendation context
        if recommendation:
            optimal_bid = recommendation.get('recommended_bid', 0)
            current_avg = df['Bid'].mean()
            
            if optimal_bid < current_avg:
                insights.append(f"**Recommended bid ({optimal_bid:.2f}) is below market average** - This suggests an aggressive strategy could be effective.")
            else:
                insights.append(f"**Recommended bid ({optimal_bid:.2f}) is above market average** - This suggests a conservative approach may be better.")
        
        # Display insights
        for insight in insights:
            st.write(insight)
    
    def create_executive_summary(self, df: pd.DataFrame, recommendation: Optional[Dict] = None, 
                               scenario_config: Optional[Dict] = None) -> str:
        """Create an executive summary of the simulation results with cost analysis."""
        
        # Get user-set market threshold (not the max bid from data)
        user_market_threshold = scenario_config.get('market_threshold', df['Bid'].max()) if scenario_config else df['Bid'].max()
        cost_price = scenario_config.get('cost_price', None) if scenario_config else None
        
        summary = f"""
        # Executive Summary
        
        ## Simulation Overview
        - **Total Rounds**: {df['Round'].max()}
        - **Number of Agents**: {len(df['Agent'].unique())}
        - **Total Bids**: {len(df)}
        - **Market Threshold**: {user_market_threshold:.2f}
        """
        
        # Add cost-based information if available
        if cost_price:
            summary += f"""
        - **Cost Price**: ${cost_price:.2f}
        - **Profit Margin Range**: {((user_market_threshold - cost_price) / cost_price * 100):.1f}% max
        """
        
        summary += f"""
        
        ## Key Performance Indicators
        - **Average Bid**: {df['Bid'].mean():.2f}
        - **Winning Bid Average**: {df[df['Winning_Bid'] == 1]['Bid'].mean():.2f}
        - **Bid Volatility**: {df['Bid'].std():.2f}
        - **Competition Intensity**: {len(df['Agent'].unique())} agents
        """
        
        # Add profitability analysis if cost data is available
        if cost_price:
            profitable_bids = df[df['Bid'] >= cost_price]
            profitability_rate = len(profitable_bids) / len(df) * 100 if len(df) > 0 else 0
            avg_profit_margin = ((df['Bid'].mean() - cost_price) / cost_price * 100) if cost_price > 0 else 0
            
            summary += f"""
        - **Cost-Covered Bids**: {profitability_rate:.1f}% of all bids
        - **Average Profit Margin**: {avg_profit_margin:.1f}%
        """
        
        summary += f"""
        
        ## Market Dynamics
        - **Bid Range**: {df['Bid'].max() - df['Bid'].min():.2f}
        - **Market Efficiency**: {(df['Bid'].max() - df['Bid'].min()) / df['Bid'].max():.1%}
        """
        
        if recommendation:
            strategic_insights = self._generate_strategic_insights(recommendation, df, scenario_config)
            summary += f"""
        ## AI Recommendation
        - **Optimal Bid**: {recommendation.get('recommended_bid', 0):.2f}
        - **Win Probability**: {recommendation.get('win_probability', 0):.1%}
        - **Confidence Level**: {recommendation.get('confidence_score', 0):.1%}
        - **Risk Assessment**: {recommendation.get('risk_level', 'Unknown')}
        """
            
            # Add profitability analysis for recommended bid
            if cost_price:
                recommended_bid = recommendation.get('recommended_bid', 0)
                profit_margin = ((recommended_bid - cost_price) / cost_price * 100) if cost_price > 0 else 0
                is_profitable = recommended_bid >= cost_price
                
                summary += f"""
        - **Profit Margin**: {profit_margin:.1f}%
        - **Profitability**: {'Profitable' if is_profitable else 'Below Cost'}
        """
            
            summary += f"""
        
        ## Strategic Insights
        {strategic_insights}
        """
        
        return summary
    
    def _generate_strategic_insights(self, recommendation: Dict, df: pd.DataFrame, 
                                   scenario_config: Optional[Dict] = None) -> str:
        """Generate strategic insights for executive summary with cost analysis."""
        
        recommended_bid = recommendation.get('recommended_bid', 0)
        win_probability = recommendation.get('win_probability', 0)
        risk_level = recommendation.get('risk_level', 'Unknown')
        
        # Calculate market insights
        avg_bid = df['Bid'].mean()
        winning_bid_avg = df[df['Winning_Bid'] == 1]['Bid'].mean()
        user_market_threshold = scenario_config.get('market_threshold', df['Bid'].max()) if scenario_config else df['Bid'].max()
        competition_intensity = len(df['Agent'].unique())
        
        # Get cost information
        cost_price = scenario_config.get('cost_price', None) if scenario_config else None
        
        # Calculate competitor aggression (lower bids = more aggressive)
        competitor_aggression = 1 - (avg_bid / user_market_threshold)
        
        # Calculate competitor consistency
        competitor_consistency = 1 - (df['Bid'].std() / df['Bid'].mean())
        
        insights = f"""
        **Strategic Analysis:**
        
        The recommended bid of **${recommended_bid:.2f}** is calculated considering both market conditions and competitor behavior. The winning bid average of **${winning_bid_avg:.2f}** suggests that {'lower' if recommended_bid < winning_bid_avg else 'higher'} bids are currently effective in the market.
        """
        
        # Add cost-based insights if available
        if cost_price:
            profit_margin = ((recommended_bid - cost_price) / cost_price * 100) if cost_price > 0 else 0
            is_profitable = recommended_bid >= cost_price
            
            insights += f"""
        
        **Cost-Based Analysis:**
        - **Cost Price**: ${cost_price:.2f}
        - **Recommended Bid Profit Margin**: {profit_margin:.1f}%
        - **Profitability Status**: {'Profitable' if is_profitable else 'Below Cost'}
        
        The recommended bid {'covers your costs' if is_profitable else 'does not cover your costs'}. {'This ensures you won\'t lose money while remaining competitive.' if is_profitable else 'Consider adjusting your cost parameters for better profitability.'}
        """
        
        insights += f"""
        
        **Market Context:**
        Given the market threshold of **${user_market_threshold:.2f}**, our recommended bid falls well within market acceptance levels.
        
        **Competition Analysis:**
        - The competition intensity at **{competition_intensity} agents** implies keen competition.
        - Given the competitor aggression level of **{competitor_aggression:.2f}**, we can anticipate {'fierce' if competitor_aggression > 0.5 else 'moderate' if competitor_aggression > 0.3 else 'conservative'} competitive bidding.
        - However, the competitor consistency of **{competitor_consistency:.2f}** suggests {'high' if competitor_consistency > 0.7 else 'moderate' if competitor_consistency > 0.4 else 'low'} level of variability in their bids.
        
        **Strategic Approach:**
        By strategically placing our bid slightly {'below' if recommended_bid < winning_bid_avg else 'above'} the winning average, we aim to enhance our chances of success while maintaining profitable margins.
        
        **Risk Assessment:**
        The {risk_level.lower()} risk level indicates that this bid strategy is {'conservative and safe' if risk_level == 'Low' else 'balanced with moderate risk' if risk_level == 'Medium' else 'aggressive with higher risk'} for the current market conditions.
        """
        
        return insights.strip()
