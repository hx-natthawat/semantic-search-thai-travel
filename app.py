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
        cache_dir = os.path.join(os.getcwd(), '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Cache directory: {cache_dir}")
        
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
def load_locations():
    try:
        logger.info(f"Loading locations from: {LOCATIONS_PATH}")
        df = pd.read_csv(LOCATIONS_PATH)
        logger.info(f"Locations loaded successfully: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading locations: {str(e)}", exc_info=True)
        st.error(f"Error loading locations: {str(e)}")
        raise e

@st.cache_data(show_spinner=True)
def load_events():
    try:
        logger.info(f"Loading events from: {EVENTS_PATH}")
        df = pd.read_csv(EVENTS_PATH)
        logger.info(f"Events loaded successfully: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading events: {str(e)}", exc_info=True)
        st.error(f"Error loading events: {str(e)}")
        raise e

@st.cache_data(show_spinner=True)
def load_articles():
    try:
        logger.info(f"Loading articles from: {ARTICLES_PATH}")
        df = pd.read_csv(ARTICLES_PATH)
        logger.info(f"Articles loaded successfully: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading articles: {str(e)}", exc_info=True)
        st.error(f"Error loading articles: {str(e)}")
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

# Semantic search functions
def search_locations(query, model, df, embeddings, top_k=TOP_K_RESULTS):
    try:
        query_embedding = model.encode(query)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
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
        return results
    except Exception as e:
        logger.error(f"Error in location search: {str(e)}", exc_info=True)
        raise e

def search_events(query, model, df, embeddings, top_k=TOP_K_RESULTS):
    try:
        query_embedding = model.encode(query)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
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
        return results
    except Exception as e:
        logger.error(f"Error in event search: {str(e)}", exc_info=True)
        raise e

def search_articles(query, model, df, embeddings, top_k=TOP_K_RESULTS):
    try:
        query_embedding = model.encode(query)
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
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
        return results
    except Exception as e:
        logger.error(f"Error in article search: {str(e)}", exc_info=True)
        raise e

def display_location_result(result):
    similarity_percentage = f"{result['similarity']*100:.1f}%"
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**ความเกี่ยวข้อง:** {similarity_percentage}")
        with col2:
            st.markdown(f"### 📍 {result['name']}")
            st.write(f"**จังหวัด:** {result['province']}")
            st.write(result['description'])
        st.write("---")

def display_event_result(result):
    similarity_percentage = f"{result['similarity']*100:.1f}%"
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**ความเกี่ยวข้อง:** {similarity_percentage}")
        with col2:
            st.markdown(f"### 🎉 {result['name']}")
            st.write(f"**สถานที่:** {result['location_name']}")
            st.write(f"**วันที่:** {result['date']}")
            st.write(f"**ประเภท:** {result['event_type']}")
            st.write(result['description'])
        st.write("---")

def display_article_result(result):
    similarity_percentage = f"{result['similarity']*100:.1f}%"
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**ความเกี่ยวข้อง:** {similarity_percentage}")
        with col2:
            st.markdown(f"### 📰 {result['title']}")
            st.write(f"**สถานที่:** {result['location_name']}")
            st.write(f"**วันที่:** {result['date']}")
            st.write(f"**ผู้เขียน:** {result['author']}")
            st.write(result['content'])
        st.write("---")

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
            
            st.info("กำลังโหลดข้อมูล...")
            locations_df = load_locations()
            events_df = load_events()
            articles_df = load_articles()
            
            st.info("กำลังเตรียมระบบค้นหา...")
            locations_embeddings = generate_embeddings(locations_df['description'].tolist(), model)
            events_embeddings = generate_embeddings(
                [f"{row['name']} {row['description']}" for _, row in events_df.iterrows()],
                model
            )
            articles_embeddings = generate_embeddings(
                [f"{row['title']} {row['content']}" for _, row in articles_df.iterrows()],
                model
            )
            st.success("พร้อมใช้งานแล้ว!")
        
        # Search interface
        query = st.text_input("🔍 ค้นหา", placeholder=SEARCH_PLACEHOLDER)
        
        if query:
            with st.spinner('กำลังค้นหา...'):
                # Perform searches
                location_results = search_locations(query, model, locations_df, locations_embeddings)
                event_results = search_events(query, model, events_df, events_embeddings)
                article_results = search_articles(query, model, articles_df, articles_embeddings)
                
                # Combine and sort results by similarity
                all_results = location_results + event_results + article_results
                all_results.sort(key=lambda x: x['similarity'], reverse=True)
                
                st.write("---")
                st.subheader("ผลการค้นหา")
                
                if not all_results:
                    st.info("ไม่พบผลการค้นหาที่ตรงกับคำค้น")
                    return
                
                # Display results by type
                for result in all_results:
                    if result['type'] == 'location':
                        display_location_result(result)
                    elif result['type'] == 'event':
                        display_event_result(result)
                    else:  # article
                        display_article_result(result)
                
    except Exception as e:
        logger.error(f"Error in main app: {str(e)}", exc_info=True)
        st.error("An error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"Application error: {str(e)}")
