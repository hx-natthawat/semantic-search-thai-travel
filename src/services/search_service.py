"""
Search service layer for handling search operations and business logic.
"""
import logging
import numpy as np
from typing import List, Dict, Any
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langdetect import detect

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, model: SentenceTransformer, top_k: int = 3):
        self.model = model
        self.top_k = top_k

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            logger.info(f"Starting to generate embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts)
            logger.info(f"Embeddings generated successfully: shape {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise

    def detect_language(self, text: str, default_language: str = "th") -> str:
        """Detect the language of input text."""
        try:
            return detect(text)
        except:
            return default_language

    def search_locations(self, query: str, df: pd.DataFrame, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Search locations based on query."""
        try:
            logger.info(f"Searching locations for query: {query}")
            query_embedding = self.model.encode(query)
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            top_indices = np.argsort(similarities)[-self.top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold for relevance
                    results.append({
                        'type': 'location',
                        'name': df.iloc[idx]['name'],
                        'description': df.iloc[idx]['description'],
                        'province': df.iloc[idx]['province'],
                        'similarity': similarities[idx]
                    })
            logger.info(f"Found {len(results)} location results")
            return results
        except Exception as e:
            logger.error(f"Error in location search: {str(e)}", exc_info=True)
            raise

    def search_events(self, query: str, df: pd.DataFrame, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Search events based on query."""
        try:
            logger.info(f"Searching events for query: {query}")
            query_embedding = self.model.encode(query)
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            top_indices = np.argsort(similarities)[-self.top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold for relevance
                    results.append({
                        'type': 'event',
                        'name': df.iloc[idx]['name'],
                        'description': df.iloc[idx]['description'],
                        'location_name': df.iloc[idx]['location_name'],
                        'date': df.iloc[idx]['date'],
                        'event_type': df.iloc[idx]['type'],
                        'similarity': similarities[idx]
                    })
            logger.info(f"Found {len(results)} event results")
            return results
        except Exception as e:
            logger.error(f"Error in event search: {str(e)}", exc_info=True)
            raise

    def search_articles(self, query: str, df: pd.DataFrame, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Search articles based on query."""
        try:
            logger.info(f"Searching articles for query: {query}")
            query_embedding = self.model.encode(query)
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            top_indices = np.argsort(similarities)[-self.top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Threshold for relevance
                    results.append({
                        'type': 'article',
                        'title': df.iloc[idx]['title'],
                        'content': df.iloc[idx]['content'],
                        'location_name': df.iloc[idx]['location_name'],
                        'date': df.iloc[idx]['date'],
                        'author': df.iloc[idx]['author'],
                        'similarity': similarities[idx]
                    })
            logger.info(f"Found {len(results)} article results")
            return results
        except Exception as e:
            logger.error(f"Error in article search: {str(e)}", exc_info=True)
            raise
