"""
UI components for the Streamlit application.
"""
import streamlit as st
from typing import Dict, Any

def display_location_result(result: Dict[str, Any], lang_config: Dict[str, Any]):
    """Display a location search result."""
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

def display_event_result(result: Dict[str, Any], lang_config: Dict[str, Any]):
    """Display an event search result."""
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

def display_article_result(result: Dict[str, Any], lang_config: Dict[str, Any]):
    """Display an article search result."""
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
