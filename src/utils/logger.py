"""
Logging configuration utility.
"""
import logging

def setup_logger():
    """Configure logging with detailed format."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)
