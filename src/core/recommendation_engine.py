import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import openai
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class BidRecommendation:
    """Data class for bid recommendations."""
    recommended_bid: float
    confidence_score: float
    win_probability: float
    reasoning: str
    risk_level: str
    alternative_bids: List[Tuple[float, float]]  # (bid, win_probability)
    market_analysis: Dict[str, float]

class OptimalBidRecommendationEngine:
    """
    Optimal Bid Recommendation Engine for Phase 3 of the roadmap.
    Analyzes simulation outcomes and provides actionable bid recommendations.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_client = None
        if openai_api_key or os.getenv("OPENAI_API_KEY"):
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            self.openai_client = openai.OpenAI(api_key=api_key)
    
    def analyze_simulation_results(self, 
                                 simulation_data: pd.DataFrame,
                                 market_threshold: float, 
                                 user_agent_name: str = "User_Agent",
                                 cost_price: float = None) -> BidRecommendation:
        """
        Analyze simulation results and generate optimal bid recommendation.
        
        Args:
            simulation_data: DataFrame with columns [Round, Agent, Bid, Winning_Bid]
            market_threshold: Maximum allowable bid
            user_agent_name: Name of the agent representing the user
            cost_price: Production/service cost price
            min_viable_bid: Minimum viable bid considering profit margin
            
        Returns:
            BidRecommendation object with optimal bid and analysis
        """
        
        # Calculate market statistics
        market_stats = self._calculate_market_statistics(simulation_data, market_threshold)
        
        # Analyze competitor behavior
        competitor_analysis = self._analyze_competitor_behavior(simulation_data, user_agent_name, market_threshold)
        
        # Generate bid recommendations with cost constraints
        bid_recommendations = self._generate_bid_recommendations(
            market_stats, competitor_analysis, market_threshold, cost_price
        )
        
        # Select optimal bid with cost constraints
        optimal_bid = self._select_optimal_bid(bid_recommendations, market_threshold, cost_price)
        
        # Generate AI explanation
        reasoning = self._generate_ai_explanation(
            optimal_bid, market_stats, competitor_analysis, simulation_data
        )
        
        # Calculate confidence and risk
        confidence_score = self._calculate_confidence_score(bid_recommendations, optimal_bid)
        win_probability = self._calculate_win_probability(optimal_bid, competitor_analysis)
        risk_level = self._assess_risk_level(optimal_bid, market_threshold, competitor_analysis)
        
        return BidRecommendation(
            recommended_bid=optimal_bid,
            confidence_score=confidence_score,
            win_probability=win_probability,
            reasoning=reasoning,
            risk_level=risk_level,
            alternative_bids=bid_recommendations[:5],  # Top 5 alternatives
            market_analysis=market_stats
        )
    
    def _calculate_market_statistics(self, data: pd.DataFrame, market_threshold: float) -> Dict[str, float]:
        """Calculate comprehensive market statistics."""
        
        stats = {
            'avg_bid': data['Bid'].mean(),
            'median_bid': data['Bid'].median(),
            'std_bid': data['Bid'].std(),
            'min_bid': data['Bid'].min(),
            'max_bid': data['Bid'].max(),
            'winning_bid_avg': data[data['Winning_Bid'] == 1]['Bid'].mean(),
            'winning_bid_median': data[data['Winning_Bid'] == 1]['Bid'].median(),
            'winning_bid_std': data[data['Winning_Bid'] == 1]['Bid'].std(),
            'bid_range': data['Bid'].max() - data['Bid'].min(),
            'threshold_utilization': data['Bid'].mean() / market_threshold,
            'competition_intensity': len(data['Agent'].unique())
        }
        
        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            stats[f'bid_percentile_{p}'] = data['Bid'].quantile(p/100)
            stats[f'winning_bid_percentile_{p}'] = data[data['Winning_Bid'] == 1]['Bid'].quantile(p/100)
        
        return stats
    
    def _analyze_competitor_behavior(self, data: pd.DataFrame, user_agent_name: str, market_threshold: float = None) -> Dict[str, any]:
        """Analyze competitor bidding patterns and strategies."""
        
        # Remove user agent from competitor analysis
        competitor_data = data[data['Agent'] != user_agent_name]
        
        analysis = {
            'competitor_agents': competitor_data['Agent'].unique().tolist(),
            'avg_competitor_bid': competitor_data['Bid'].mean(),
            'competitor_bid_std': competitor_data['Bid'].std(),
            'competitor_winning_bid_avg': competitor_data[competitor_data['Winning_Bid'] == 1]['Bid'].mean(),
            'competitor_aggression_level': self._calculate_aggression_level(competitor_data, market_threshold),
            'competitor_consistency': self._calculate_consistency(competitor_data),
            'competitor_learning_rate': self._calculate_learning_rate(competitor_data)
        }
        
        # Analyze individual competitor strategies
        competitor_strategies = {}
        for agent in competitor_data['Agent'].unique():
            agent_data = competitor_data[competitor_data['Agent'] == agent]
            competitor_strategies[agent] = {
                'avg_bid': agent_data['Bid'].mean(),
                'win_rate': (agent_data['Winning_Bid'] == 1).mean(),
                'bid_volatility': agent_data['Bid'].std(),
                'strategy_type': self._classify_agent_strategy(agent_data)
            }
        
        analysis['competitor_strategies'] = competitor_strategies
        
        return analysis
    
    def _calculate_aggression_level(self, data: pd.DataFrame, market_threshold: float = None) -> float:
        """Calculate how aggressive competitors are (lower bids = more aggressive)."""
        avg_bid = data['Bid'].mean()
        # Use actual market threshold if provided, otherwise use max bid as fallback
        threshold = market_threshold if market_threshold else data['Bid'].max()
        return 1 - (avg_bid / threshold)  # Higher value = more aggressive
    
    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calculate how consistent competitor bids are."""
        return 1 - (data['Bid'].std() / data['Bid'].mean())  # Lower std = more consistent
    
    def _calculate_learning_rate(self, data: pd.DataFrame) -> float:
        """Calculate how quickly competitors are learning (improving)."""
        if len(data) < 10:
            return 0.5  # Default if not enough data
        
        # Split data into early and late periods
        mid_point = len(data) // 2
        early_bids = data.iloc[:mid_point]['Bid'].mean()
        late_bids = data.iloc[mid_point:]['Bid'].mean()
        
        # Lower bids over time indicate learning
        if early_bids > late_bids:
            return (early_bids - late_bids) / early_bids
        else:
            return 0.0
    
    def _classify_agent_strategy(self, agent_data: pd.DataFrame) -> str:
        """Classify an agent's bidding strategy."""
        avg_bid = agent_data['Bid'].mean()
        volatility = agent_data['Bid'].std()
        win_rate = (agent_data['Winning_Bid'] == 1).mean()
        
        if avg_bid < agent_data['Bid'].quantile(0.25):
            return "Aggressive"
        elif avg_bid > agent_data['Bid'].quantile(0.75):
            return "Conservative"
        elif volatility > agent_data['Bid'].std() * 1.5:
            return "Volatile"
        elif win_rate > 0.3:
            return "Strategic"
        else:
            return "Balanced"
    
    def _generate_bid_recommendations(self, 
                                    market_stats: Dict[str, float],
                                    competitor_analysis: Dict[str, any],
                                    market_threshold: float,
                                    cost_price: float = None) -> List[Tuple[float, float]]:
        """Generate multiple bid recommendations with win probabilities, considering cost constraints."""
        
        recommendations = []
        
        # Strategy 1: Beat average competitor bid by 5%
        competitor_avg = competitor_analysis['avg_competitor_bid']
        bid_1 = competitor_avg * 0.95
        if cost_price and bid_1 < cost_price:
            bid_1 = cost_price  # Ensure cost coverage
        prob_1 = self._estimate_win_probability(bid_1, competitor_analysis)
        recommendations.append((bid_1, prob_1))
        
        # Strategy 2: Target winning bid average
        winning_avg = market_stats['winning_bid_avg']
        bid_2 = winning_avg * 0.98  # Slightly below average winning bid
        if cost_price and bid_2 < cost_price:
            bid_2 = cost_price  # Ensure cost coverage
        prob_2 = self._estimate_win_probability(bid_2, competitor_analysis)
        recommendations.append((bid_2, prob_2))
        
        # Strategy 3: Conservative approach (25th percentile)
        bid_3 = market_stats['bid_percentile_25']
        if cost_price and bid_3 < cost_price:
            bid_3 = cost_price  # Ensure cost coverage
        prob_3 = self._estimate_win_probability(bid_3, competitor_analysis)
        recommendations.append((bid_3, prob_3))
        
        # Strategy 4: Aggressive approach (10th percentile)
        bid_4 = market_stats['bid_percentile_10']
        if cost_price and bid_4 < cost_price:
            bid_4 = cost_price  # Ensure cost coverage
        prob_4 = self._estimate_win_probability(bid_4, competitor_analysis)
        recommendations.append((bid_4, prob_4))
        
        # Strategy 5: Market threshold optimization
        bid_5 = market_threshold * 0.85  # 15% below threshold
        if cost_price and bid_5 < cost_price:
            bid_5 = cost_price  # Ensure cost coverage
        prob_5 = self._estimate_win_probability(bid_5, competitor_analysis)
        recommendations.append((bid_5, prob_5))
        
        # Strategy 6: Cost-based competitive bid
        if cost_price:
            # Target 15% above cost for competitive positioning
            target_margin = 0.15
            bid_6 = cost_price * (1 + target_margin)
            if bid_6 <= market_threshold:
                prob_6 = self._estimate_win_probability(bid_6, competitor_analysis)
                recommendations.append((bid_6, prob_6))
        
        # Strategy 7: Cost floor strategy
        if cost_price:
            bid_7 = cost_price
            prob_7 = self._estimate_win_probability(bid_7, competitor_analysis)
            recommendations.append((bid_7, prob_7))
        
        # Strategy 8: Sweet spot strategy (based on simulation learning)
        if cost_price and market_stats['winning_bid_avg']:
            # Target slightly above the average winning bid for optimal profit + win probability
            sweet_spot_bid = market_stats['winning_bid_avg'] * 1.02  # 2% above winning average
            if sweet_spot_bid <= market_threshold and sweet_spot_bid >= cost_price:
                prob_8 = self._estimate_win_probability(sweet_spot_bid, competitor_analysis)
                recommendations.append((sweet_spot_bid, prob_8))
        
        # Strategy 9: Competitive margin strategy
        if cost_price:
            # Target 5-10% profit margin for competitive positioning
            competitive_margin = 0.08  # 8% profit margin
            competitive_bid = cost_price * (1 + competitive_margin)
            if competitive_bid <= market_threshold:
                prob_9 = self._estimate_win_probability(competitive_bid, competitor_analysis)
                recommendations.append((competitive_bid, prob_9))
        
        # Don't sort by win probability - let the scoring function decide the optimal bid
        # recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations
    
    def _estimate_win_probability(self, bid: float, competitor_analysis: Dict[str, any]) -> float:
        """Estimate probability of winning with a given bid."""
        
        # Enhanced probability model based on competitor behavior
        competitor_avg = competitor_analysis['avg_competitor_bid']
        competitor_std = competitor_analysis['competitor_bid_std']
        
        # Calculate how many standard deviations below average our bid is
        z_score = (competitor_avg - bid) / max(competitor_std, 1)
        
        # Use sigmoid function for more realistic probability distribution
        # This creates better differentiation between bids
        win_prob = 1 / (1 + np.exp(-z_score + 1))  # Shift by 1 to favor competitive bids
        
        # Ensure reasonable bounds
        win_prob = min(0.95, max(0.05, win_prob))
        
        return win_prob
    
    def _select_optimal_bid(self, recommendations: List[Tuple[float, float]], 
                          market_threshold: float,
                          cost_price: float = None) -> float:
        """Select the optimal bid from recommendations, considering cost constraints."""
        
        if not recommendations:
            if cost_price:
                return cost_price  # Return cost price as fallback
            return market_threshold * 0.7  # Default fallback
        
        # Use a weighted approach considering both win probability and profitability
        best_score = 0
        optimal_bid = recommendations[0][0]
        
        print(f"🔍 Analyzing {len(recommendations)} bid strategies:")
        
        for i, (bid, win_prob) in enumerate(recommendations):
            # Enhanced scoring that considers cost-based profitability
            if cost_price:
                # Calculate profit and profit margin
                profit = bid - cost_price
                profit_margin = profit / cost_price if cost_price > 0 else 0
                
                # NEW SCORING: Balance win probability and profit margin
                # Remove market_position_factor bias that favors lower bids
                # Focus on win probability and profit margin balance
                
                # Normalize profit margin to 0-1 scale (0% to 20% profit margin)
                normalized_profit = min(1.0, profit_margin / 0.20)  # Cap at 20% profit margin
                
                # Score = 60% win probability + 40% profit margin
                # This ensures we don't just pick the lowest bid
                score = 0.6 * win_prob + 0.4 * normalized_profit
            else:
                # Fallback to original scoring
                score = win_prob * (1 - bid / market_threshold)
            
            print(f"   Strategy {i+1}: Bid=${bid:.2f}, WinProb={win_prob:.3f}, Score={score:.4f}")
            
            if score > best_score:
                best_score = score
                optimal_bid = bid
                print(f"   🎯 NEW BEST: Bid=${bid:.2f}, Score={score:.4f}")
        
        print(f"🏆 Final recommendation: ${optimal_bid:.2f} with score {best_score:.4f}")
        
        return optimal_bid
    
    def _calculate_confidence_score(self, recommendations: List[Tuple[float, float]], 
                                  optimal_bid: float) -> float:
        """Calculate confidence score for the recommendation."""
        
        if not recommendations:
            return 0.5
        
        # Find the recommendation closest to optimal bid
        closest_rec = min(recommendations, key=lambda x: abs(x[0] - optimal_bid))
        
        # Confidence based on win probability and consistency of recommendations
        win_prob = closest_rec[1]
        consistency = 1 - (np.std([r[0] for r in recommendations]) / np.mean([r[0] for r in recommendations]))
        
        return (win_prob + consistency) / 2
    
    def _calculate_win_probability(self, bid: float, competitor_analysis: Dict[str, any]) -> float:
        """Calculate win probability for the recommended bid."""
        return self._estimate_win_probability(bid, competitor_analysis)
    
    def _assess_risk_level(self, bid: float, market_threshold: float, 
                          competitor_analysis: Dict[str, any]) -> str:
        """Assess the risk level of the recommended bid."""
        
        # Calculate risk factors
        margin_risk = bid / market_threshold  # Higher = more risk
        competition_risk = competitor_analysis['competitor_aggression_level']  # Higher = more risk
        volatility_risk = 1 - competitor_analysis['competitor_consistency']  # Higher = more risk
        
        total_risk = (margin_risk + competition_risk + volatility_risk) / 3
        
        if total_risk < 0.3:
            return "Low"
        elif total_risk < 0.6:
            return "Medium"
        else:
            return "High"
    
    def _generate_ai_explanation(self, 
                               optimal_bid: float,
                               market_stats: Dict[str, float],
                               competitor_analysis: Dict[str, any],
                               simulation_data: pd.DataFrame) -> str:
        """Generate AI-powered explanation for the recommendation."""
        
        if self.openai_client is None:
            return self._generate_simple_explanation(optimal_bid, market_stats, competitor_analysis)
        
        try:
            # Prepare context for AI
            context = f"""
            Market Analysis:
            - Average bid: {market_stats['avg_bid']:.2f}
            - Winning bid average: {market_stats['winning_bid_avg']:.2f}
            - Market threshold: {simulation_data['Bid'].max():.2f}
            - Competition intensity: {market_stats['competition_intensity']} agents
            
            Competitor Analysis:
            - Average competitor bid: {competitor_analysis['avg_competitor_bid']:.2f}
            - Competitor aggression level: {competitor_analysis['competitor_aggression_level']:.2f}
            - Competitor consistency: {competitor_analysis['competitor_consistency']:.2f}
            
            Recommended Bid: {optimal_bid:.2f}
            
            Please provide a concise, professional explanation of why this bid is recommended, 
            considering the market conditions and competitor behavior.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert procurement consultant providing bid recommendations. Be concise and professional."},
                    {"role": "user", "content": context}
                ],
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ AI explanation error: {e}")
            return self._generate_simple_explanation(optimal_bid, market_stats, competitor_analysis)
    
    def _generate_simple_explanation(self, 
                                   optimal_bid: float,
                                   market_stats: Dict[str, float],
                                   competitor_analysis: Dict[str, any]) -> str:
        """Generate a simple explanation without AI."""
        
        explanation = f"""
        Based on the simulation analysis, a bid of {optimal_bid:.2f} is recommended.
        
        Key factors:
        • Market average: {market_stats['avg_bid']:.2f}
        • Winning bid average: {market_stats['winning_bid_avg']:.2f}
        • Competitor average: {competitor_analysis['avg_competitor_bid']:.2f}
        • Competition level: {competitor_analysis['competitor_aggression_level']:.1%} aggressive
        
        This bid balances winning probability with profit margin based on observed competitor behavior.
        """
        
        return explanation.strip()
    
    def generate_comprehensive_report(self, recommendation: BidRecommendation) -> Dict[str, any]:
        """Generate a comprehensive report for the recommendation."""
        
        return {
            'recommendation': {
                'optimal_bid': recommendation.recommended_bid,
                'confidence_score': recommendation.confidence_score,
                'win_probability': recommendation.win_probability,
                'risk_level': recommendation.risk_level,
                'reasoning': recommendation.reasoning
            },
            'alternatives': [
                {'bid': bid, 'win_probability': prob} 
                for bid, prob in recommendation.alternative_bids
            ],
            'market_analysis': recommendation.market_analysis,
            'summary': {
                'recommendation_type': 'Optimal Bid Strategy',
                'confidence_level': 'High' if recommendation.confidence_score > 0.7 else 'Medium' if recommendation.confidence_score > 0.4 else 'Low',
                'risk_assessment': recommendation.risk_level,
                'expected_outcome': 'Favorable' if recommendation.win_probability > 0.6 else 'Competitive' if recommendation.win_probability > 0.4 else 'Challenging'
            }
        }
