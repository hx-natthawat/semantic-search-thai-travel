"""
Main application entry point.
"""
import os
import streamlit as st
from config import *
from api.model_api import ModelAPI
from services.search_service import SearchService
from repositories.data_repository import DataRepository
from ui.components import (
    display_location_result,
    display_event_result,
    display_article_result
)
from utils.logger import setup_logger

# Set up logging
logger = setup_logger()

# Get project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize services
@st.cache_resource(show_spinner=True)
def initialize_services():
    """Initialize all required services."""
    try:
        # Initialize model
        model = ModelAPI.load_model(MODEL_NAME, PROJECT_ROOT)
        
        # Initialize services
        search_service = SearchService(model, TOP_K_RESULTS)
        data_repository = DataRepository(PROJECT_ROOT)
        
        return model, search_service, data_repository
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}", exc_info=True)
        raise

def main():
    try:
        # Set page config
        st.set_page_config(
            page_title=SUPPORTED_LANGUAGES[DEFAULT_LANGUAGE]["app_title"],
            page_icon=PAGE_ICON,
            layout=LAYOUT
        )

        # Initialize services
        model, search_service, data_repository = initialize_services()
        
        # Language selection
        languages = {lang: config["name"] for lang, config in SUPPORTED_LANGUAGES.items()}
        selected_language = st.sidebar.selectbox(
            "Language / ภาษา",
            options=list(languages.keys()),
            format_func=lambda x: languages[x],
            index=list(languages.keys()).index(DEFAULT_LANGUAGE)
        )
        
        lang_config = SUPPORTED_LANGUAGES[selected_language]
        
        # Display header
        st.title(lang_config["app_title"])
        st.write(lang_config["app_description"])
        
        # Load data
        locations_df = data_repository.load_locations(selected_language, lang_config["locations_path"])
        events_df = data_repository.load_events(EVENTS_PATH)
        articles_df = data_repository.load_articles(ARTICLES_PATH)
        
        # Generate embeddings
        locations_embeddings = search_service.generate_embeddings(locations_df['description'].tolist())
        events_embeddings = search_service.generate_embeddings(
            [f"{row['name']} {row['description']}" for _, row in events_df.iterrows()]
        )
        articles_embeddings = search_service.generate_embeddings(
            [f"{row['title']} {row['content']}" for _, row in articles_df.iterrows()]
        )
        
        # Search interface
        query = st.text_input("🔍", placeholder=lang_config["search_placeholder"], key="search_input")
        
        if query:
            logger.info(f"Processing search query: {query}")
            
            # Auto-detect query language
            detected_lang = search_service.detect_language(query, DEFAULT_LANGUAGE)
            if detected_lang != selected_language and detected_lang in SUPPORTED_LANGUAGES:
                st.info(f"Detected {SUPPORTED_LANGUAGES[detected_lang]['name']} query, searching in {SUPPORTED_LANGUAGES[detected_lang]['name']} database...")
                locations_df = data_repository.load_locations(detected_lang, SUPPORTED_LANGUAGES[detected_lang]["locations_path"])
                locations_embeddings = search_service.generate_embeddings(locations_df['description'].tolist())
            
            # Perform searches
            location_results = search_service.search_locations(query, locations_df, locations_embeddings)
            event_results = search_service.search_events(query, events_df, events_embeddings)
            article_results = search_service.search_articles(query, articles_df, articles_embeddings)
            
            # Combine and sort results
            all_results = location_results + event_results + article_results
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Total results found: {len(all_results)}")
            
            # Display results
            if all_results:
                st.write("---")
                st.subheader(lang_config["results_header"])
                
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
