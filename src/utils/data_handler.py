import os
import pandas as pd
import logging
from pathlib import Path
from src.utils.config import Config  # Import config to get the correct path

#  Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataHandler:
    """Handles data storage and retrieval for bidding simulation."""

    @staticmethod
    def save_bid_data(round_num, bids, winning_bid):
        """Stores bid data in a single CSV file."""
        config = Config.load_config()
        data_file = config["DATA_FILE"]  #  Ensure the same file path is used

        # Ensure `data/` directory exists before saving
        Path(os.path.dirname(data_file)).mkdir(parents=True, exist_ok=True)

        data = [{
            "Round": int(round_num),  # Ensure Round is always an integer
            "Agent": agent,
            "Bid": round(float(bid), 4),  #  Round bids to 4 decimal places
            "Winning_Bid": bid == winning_bid
        } for agent, bid in bids.items()]
        
        df = pd.DataFrame(data)

        # Check if the CSV file is empty to add headers
        file_exists = Path(data_file).exists() and os.path.getsize(data_file) > 0

        try:
            df.to_csv(data_file, mode='a', header=not file_exists, index=False)
            logging.info(f"Successfully saved bid data for Round {round_num}.")
        except Exception as e:
            logging.error(f" Error saving bid data: {e}")

    @staticmethod
    def load_bid_data():
        """Loads bid data from a CSV file for analysis."""
        config = Config.load_config()
        data_file = config["DATA_FILE"]  # Ensure correct file path
        file_path = Path(data_file)

        if file_path.exists() and file_path.stat().st_size > 0:
            try:
                df = pd.read_csv(data_file)

                # Convert "Round" column to integer
                if "Round" in df.columns:
                    df["Round"] = pd.to_numeric(df["Round"], errors='coerce').fillna(0).astype(int)

                # Convert "Bid" to float safely
                if "Bid" in df.columns:
                    df["Bid"] = pd.to_numeric(df["Bid"], errors='coerce')

                logging.info(" Bid data loaded successfully.")
                return df
            except Exception as e:
                logging.error(f"⚠️ Error loading bid data: {e}")
                return pd.DataFrame(columns=["Round", "Agent", "Bid", "Winning_Bid"])
        else:
            logging.warning("⚠️ No bid data found. Returning an empty DataFrame.")
            return pd.DataFrame(columns=["Round", "Agent", "Bid", "Winning_Bid"])

if __name__ == "__main__":
    test_bids = {"Agent 1": 95.2356, "Agent 2": 100.5678, "Agent 3": 98.345}
    winning_bid = min(test_bids.values())

    DataHandler.save_bid_data(round_num=1, bids=test_bids, winning_bid=winning_bid)
    df = DataHandler.load_bid_data()
    print(df.head())
