"""
Data repository layer for handling data access operations.
"""
import os
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataRepository:
    def __init__(self, project_root: str):
        self.project_root = project_root

    def load_locations(self, language: str, locations_path: str) -> pd.DataFrame:
        """Load location data for a specific language."""
        try:
            path = os.path.join(self.project_root, locations_path)
            logger.info(f"Loading {language} locations from: {path}")
            df = pd.read_csv(path)
            logger.info(f"Locations loaded successfully: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading locations: {str(e)}", exc_info=True)
            raise

    def load_events(self, events_path: str) -> pd.DataFrame:
        """Load events data."""
        try:
            path = os.path.join(self.project_root, events_path)
            logger.info(f"Loading events from: {path}")
            df = pd.read_csv(path)
            logger.info(f"Events loaded successfully: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading events: {str(e)}", exc_info=True)
            raise

    def load_articles(self, articles_path: str) -> pd.DataFrame:
        """Load articles data."""
        try:
            path = os.path.join(self.project_root, articles_path)
            logger.info(f"Loading articles from: {path}")
            df = pd.read_csv(path)
            logger.info(f"Articles loaded successfully: {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading articles: {str(e)}", exc_info=True)
            raise
