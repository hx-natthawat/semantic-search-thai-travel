import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
import logging
import sys
import os
from config import *

# Set up project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
try:
    st.set_page_config(
        page_title=SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["app_title"],
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
        cache_dir = os.path.join(PROJECT_ROOT, '.cache')
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
def load_locations(language):
    try:
        locations_path = os.path.join(PROJECT_ROOT, SUPPORTED_LANGUAGES[language]["locations_path"])
        logger.info(f"Loading {language} locations from: {locations_path}")
        df = pd.read_csv(locations_path)
        logger.info(f"Locations loaded successfully: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading locations: {str(e)}", exc_info=True)
        st.error(f"Error loading locations: {str(e)}")
        raise e

@st.cache_data(show_spinner=True)
def load_events():
    try:
        events_path = os.path.join(PROJECT_ROOT, EVENTS_PATH)
        logger.info(f"Loading events from: {events_path}")
        df = pd.read_csv(events_path)
        logger.info(f"Events loaded successfully: {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading events: {str(e)}", exc_info=True)
        st.error(f"Error loading events: {str(e)}")
        raise e

@st.cache_data(show_spinner=True)
def load_articles():
    try:
        articles_path = os.path.join(PROJECT_ROOT, ARTICLES_PATH)
        logger.info(f"Loading articles from: {articles_path}")
        df = pd.read_csv(articles_path)
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

def detect_language(text):
    try:
        return detect(text)
    except:
        return DEFAULT_LANGUAGE

# Semantic search functions
def search_locations(query, model, df, embeddings, lang_config, top_k=TOP_K_RESULTS):
    try:
        logger.info(f"Searching locations for query: {query}")
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
        logger.info(f"Found {len(results)} location results")
        return results
    except Exception as e:
        logger.error(f"Error in location search: {str(e)}", exc_info=True)
        raise e

def search_events(query, model, df, embeddings, lang_config, top_k=TOP_K_RESULTS):
    try:
        logger.info(f"Searching events for query: {query}")
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
        logger.info(f"Found {len(results)} event results")
        return results
    except Exception as e:
        logger.error(f"Error in event search: {str(e)}", exc_info=True)
        raise e

def search_articles(query, model, df, embeddings, lang_config, top_k=TOP_K_RESULTS):
    try:
        logger.info(f"Searching articles for query: {query}")
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
        logger.info(f"Found {len(results)} article results")
        return results
    except Exception as e:
        logger.error(f"Error in article search: {str(e)}", exc_info=True)
        raise e

def display_location_result(result, lang_config):
    similarity_percentage = f"{result['similarity']*100:.1f}%"
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**{lang_config['relevance']}:** {similarity_percentage}")
        with col2:
            st.markdown(f"### 📍 {result['name']}")
            st.write(f"**{lang_config['province']}:** {result['province']}")
            st.write(result['description'])
        st.write("---")

def display_event_result(result, lang_config):
    similarity_percentage = f"{result['similarity']*100:.1f}%"
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**{lang_config['relevance']}:** {similarity_percentage}")
        with col2:
            st.markdown(f"### 🎉 {result['name']}")
            st.write(f"**{lang_config['location']}:** {result['location_name']}")
            st.write(f"**{lang_config['date']}:** {result['date']}")
            st.write(f"**{lang_config['type']}:** {result['event_type']}")
            st.write(result['description'])
        st.write("---")

def display_article_result(result, lang_config):
    similarity_percentage = f"{result['similarity']*100:.1f}%"
    with st.container():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**{lang_config['relevance']}:** {similarity_percentage}")
        with col2:
            st.markdown(f"### 📰 {result['title']}")
            st.write(f"**{lang_config['location']}:** {result['location_name']}")
            st.write(f"**{lang_config['date']}:** {result['date']}")
            st.write(f"**{lang_config['author']}:** {result['author']}")
            st.write(result['content'])
        st.write("---")

# Main app
def main():
    try:
        # Language selection
        languages = {lang: config["name"] for lang, config in SUPPORTED_LANGUAGES.items()}
        selected_language = st.sidebar.selectbox(
            "Language / ภาษา",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            index=list(languages.keys()).index(DEFAULT_LANGUAGE)
        )
        
        lang_config = SUPPORTED_LANGUAGES[selected_language]
        
        st.title(lang_config["app_title"])
        st.write(lang_config["app_description"])
        
        # Load model and data
        model = load_model()
        locations_df = load_locations(selected_language)
        events_df = load_events()
        articles_df = load_articles()
        
        # Generate embeddings
        locations_embeddings = generate_embeddings(locations_df['description'].tolist(), model)
        events_embeddings = generate_embeddings(
            [f"{row['name']} {row['description']}" for _, row in events_df.iterrows()],
            model
        )
        articles_embeddings = generate_embeddings(
            [f"{row['title']} {row['content']}" for _, row in articles_df.iterrows()],
            model
        )
        
        # Search interface
        query = st.text_input("🔍", placeholder=lang_config["search_placeholder"], key="search_input")
        
        if query:
            logger.info(f"Processing search query: {query}")
            
            # Auto-detect query language if different from UI language
            detected_lang = detect_language(query)
            if detected_lang != selected_language and detected_lang in SUPPORTED_LANGUAGES:
                st.info(f"Detected {SUPPORTED_LANGUAGES[detected_lang]['name']} query, searching in {SUPPORTED_LANGUAGES[detected_lang]['name']} database...")
                locations_df = load_locations(detected_lang)
                locations_embeddings = generate_embeddings(locations_df['description'].tolist(), model)
            
            # Perform searches
            location_results = search_locations(query, model, locations_df, locations_embeddings, lang_config)
            event_results = search_events(query, model, events_df, events_embeddings, lang_config)
            article_results = search_articles(query, model, articles_df, articles_embeddings, lang_config)
            
            # Combine and sort results by similarity
            all_results = location_results + event_results + article_results
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Total results found: {len(all_results)}")
            
            if all_results:
                st.write("---")
                st.subheader(lang_config["results_header"])
                
                # Display results by type
                for result in all_results:
                    if result['type'] == 'location':
                        display_location_result(result, lang_config)
                    elif result['type'] == 'event':
                        display_event_result(result, lang_config)
                    else:  # article
                        display_article_result(result, lang_config)
            else:
                st.info(lang_config["no_results"])

    except Exception as e:
        logger.error(f"Error in main app: {str(e)}", exc_info=True)
        st.error("An error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"Application error: {str(e)}")
