"""
API layer for model interactions.
"""
import os
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ModelAPI:
    @staticmethod
    def load_model(model_name: str, project_root: str) -> SentenceTransformer:
        """Load and initialize the SentenceTransformer model."""
        try:
            logger.info(f"Starting to load model: {model_name}")
            cache_dir = os.path.join(project_root, '.cache')
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Cache directory: {cache_dir}")
            
            model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                trust_remote_code=True
            )
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
