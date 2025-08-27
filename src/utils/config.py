import os
import json
import pandas as pd
from dotenv import load_dotenv

class Config:
    """Handles configuration settings for the bidding system."""
    CONFIG_FILE = "config.json"
    DEFAULT_CONFIG = {
        "OPENAI_API_KEY": None,  # Will be loaded from .env if available
        "BIDDING_ROUNDS": 20,
        "INITIAL_THRESHOLD": 100,
        "DATA_FILE": "data/bid_history.csv"  #  Ensure single storage location
    }

    @staticmethod
    def load_config():
        """Loads configuration from a JSON file or creates a default one."""
        load_dotenv()  # Load .env variables
        config = Config.DEFAULT_CONFIG.copy()  # Start with default settings

        if os.path.exists(Config.CONFIG_FILE):
            try:
                with open(Config.CONFIG_FILE, "r") as file:
                    user_config = json.load(file)
                    config.update(user_config)  # Merge user config
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️ Config file error: {e}, loading defaults.")

        # Overwrite OpenAI API Key from .env (for security)
        env_api_key = os.getenv("OPENAI_API_KEY")
        if env_api_key:
            config["OPENAI_API_KEY"] = env_api_key

        return config

    @staticmethod
    def save_config(config):
        """Saves configuration settings to a JSON file."""
        with open(Config.CONFIG_FILE, "w") as file:
            json.dump(config, file, indent=4)
        print(" Configuration saved successfully.")

if __name__ == "__main__":
    config = Config.load_config()
    print("Loaded Configuration:", config)
