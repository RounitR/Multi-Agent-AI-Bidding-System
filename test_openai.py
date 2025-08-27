import openai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded
if not api_key:
    print("OpenAI API Key not found! Please check your .env file.")
else:
    print(" OpenAI API Key Loaded Successfully!")

# Make a test API call using the latest OpenAI format
try:
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello AI, how are you?"}]
    )
    
    print(" OpenAI API is working!")
    print("AI Response:", response.choices[0].message.content)
except Exception as e:
    print("OpenAI API call failed:", str(e))
