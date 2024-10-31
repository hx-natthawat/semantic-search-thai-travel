import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys
import os
from config import *

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
try:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=PAGE_ICON,
        layout=LAYOUT
    )
except Exception as e:
    logger.error(f"Error in page config: {str(e)}")
    st.error(f"Error in page config: {str(e)}")

# Initialize the model
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        logger.info(f"Starting to load model: {MODEL_NAME}")
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.getcwd(), '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Cache directory: {cache_dir}")
        
        # Load the model with explicit cache directory
        model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=cache_dir,
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        st.error(f"Error loading model: {str(e)}")
        raise e

# Load and cache data
@st.cache_data(show_spinner=True)
def load_data():
    try:
        logger.info(f"Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Data loaded successfully: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        st.error(f"Error loading data: {str(e)}")
        raise e

# Generate embeddings
@st.cache_data(show_spinner=True)
def generate_embeddings(texts, _model):
    try:
        logger.info(f"Starting to generate embeddings for {len(texts)} texts")
        embeddings = _model.encode(texts)
        logger.info(f"Embeddings generated successfully: shape {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        st.error(f"Error generating embeddings: {str(e)}")
        raise e

# Semantic search function
def semantic_search(query, model, df, embeddings, top_k=TOP_K_RESULTS):
    try:
        logger.info(f"Performing semantic search for query: {query}")
        query_embedding = model.encode(query)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'name': df.iloc[idx]['name'],
                'description': df.iloc[idx]['description'],
                'province': df.iloc[idx]['province'],
                'similarity': similarities[idx]
            })
        logger.info(f"Search completed: found {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
        st.error(f"Error in semantic search: {str(e)}")
        raise e

# Main app
def main():
    try:
        st.title(APP_TITLE)
        st.write(APP_DESCRIPTION)
        
        # Load model and data with progress messages
        with st.spinner('กำลังโหลดโมเดลและข้อมูล...'):
            logger.info("Starting main application")
            st.info("กำลังโหลดโมเดล AI...")
            model = load_model()
            
            st.info("กำลังโหลดข้อมูลสถานที่ท่องเที่ยว...")
            df = load_data()
            
            st.info("กำลังเตรียมระบบค้นหา...")
            embeddings = generate_embeddings(df['description'].tolist(), model)
            st.success("พร้อมใช้งานแล้ว!")
        
        # Search interface
        query = st.text_input("🔍 ค้นหาสถานที่ท่องเที่ยว", placeholder=SEARCH_PLACEHOLDER)
        
        if query:
            with st.spinner('กำลังค้นหา...'):
                results = semantic_search(query, model, df, embeddings)
            
            st.write("---")
            st.subheader("ผลการค้นหา")
            
            for idx, result in enumerate(results, 1):
                similarity_percentage = f"{result['similarity']*100:.1f}%"
                
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**ความเกี่ยวข้อง:** {similarity_percentage}")
                    with col2:
                        st.markdown(f"### {result['name']}")
                        st.write(f"**จังหวัด:** {result['province']}")
                        st.write(result['description'])
                    st.write("---")
    except Exception as e:
        logger.error(f"Error in main app: {str(e)}", exc_info=True)
        st.error("An error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"Application error: {str(e)}")
