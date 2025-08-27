import os
import random
import numpy as np
import openai
from dotenv import load_dotenv

# Load OpenAI API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ WARNING: OpenAI API Key not found. AI-powered threshold prediction will be disabled.")
else:
    openai.api_key = OPENAI_API_KEY  # Set OpenAI API Key globally


def dynamic_market_threshold(current_threshold, all_bids):
    """
    Adjusts the market threshold dynamically using AI + traditional statistical analysis.
    
    - Uses AI to analyze bidding trends.
    - Adds controlled randomness for realistic fluctuations.
    - Prevents extreme dips in market threshold.
    """

    avg_bid = np.mean(all_bids)
    std_dev = np.std(all_bids)  #  Consider bid volatility
    fluctuation = random.uniform(-3, 3)  # Minor random fluctuation

    #  Step 1: Traditional Statistical Adjustment
    if avg_bid < current_threshold * 0.85:
        new_threshold = current_threshold * 0.97 + fluctuation
    elif avg_bid > current_threshold * 1.1:
        new_threshold = current_threshold * 1.03 + fluctuation
    else:
        new_threshold = current_threshold + fluctuation

    # Step 2: AI-Powered Market Prediction (If API is Available)
    ai_adjustment = get_ai_market_adjustment(current_threshold, avg_bid, std_dev)
    if ai_adjustment:
        new_threshold = (new_threshold + ai_adjustment) / 2  # Hybrid AI + Statistical adjustment

    # Step 3: Ensure Threshold Stability
    return max(500, new_threshold)  # Market threshold cannot drop below 500


def get_ai_market_adjustment(current_threshold, avg_bid, std_dev):
    """
    Uses OpenAI GPT-4 to analyze bid trends and suggest threshold adjustments.
    """

    if not OPENAI_API_KEY:
        return None  # Skip AI adjustment if API Key is missing

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI market analyst. Respond with ONLY a number representing the suggested market threshold. Do not include any text, explanations, or other characters - just the number."},
                {"role": "user", "content": f"""
                The current market threshold is {current_threshold}.
                - Average bid: {avg_bid}
                - Standard deviation of bids: {std_dev}
                
                Based on these trends, suggest a new market threshold that ensures fair pricing and prevents drastic fluctuations.
                Respond with ONLY a number.
                """}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract number from response
        try:
            import re
            number_match = re.search(r'\d+\.?\d*', content)
            if number_match:
                suggested_threshold = float(number_match.group())
            else:
                suggested_threshold = float(content)
            
            print(f"🤖 AI-Suggested Market Threshold: {suggested_threshold}")
            return suggested_threshold
        except (ValueError, TypeError):
            print(f"⚠️ Could not parse AI market threshold: '{content}'. Using statistical adjustment only.")
            return None
    except Exception as e:
        print(f"⚠️ OpenAI API Error: {e}")
        return None


#  Example Usage
if __name__ == "__main__":
    current_threshold = 1000
    all_bids = np.random.uniform(800, 1200, 50)  # Simulating 50 recent bids

    new_market_threshold = dynamic_market_threshold(current_threshold, all_bids)
    print(f"\n New Adjusted Market Threshold: {new_market_threshold}")
