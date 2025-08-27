import logging
import os

#  Ensure logs directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

#  Configure Logger
LOG_FILE = os.path.join(LOG_DIR, "bidding_system.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Save logs to file
        logging.StreamHandler()  # Print logs to console
    ]
)

# Logger instance
logger = logging.getLogger("BiddingSystem")

#  Example function for logging warnings/errors
def log_error(error_msg):
    """Logs error messages."""
    logger.error(f" ERROR: {error_msg}")

def log_info(info_msg):
    """Logs informational messages."""
    logger.info(f"INFO: {info_msg}")

def log_debug(debug_msg):
    """Logs debug messages."""
    logger.debug(f"DEBUG: {debug_msg}")

if __name__ == "__main__":
    log_info("Logger initialized successfully!")
    log_debug("This is a debug message")
    log_error("This is an error message")
