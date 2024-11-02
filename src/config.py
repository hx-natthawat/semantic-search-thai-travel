"""
Configuration settings for the Multilingual Travel Semantic Search application.
"""

# Model settings
MODEL_NAME = "jinaai/jina-embeddings-v3"
TOP_K_RESULTS = 3

# Data settings
THAI_LOCATIONS_PATH = "data/thai_locations.csv"
ENGLISH_LOCATIONS_PATH = "data/english_locations.csv"
EVENTS_PATH = "data/events.csv"
ARTICLES_PATH = "data/articles.csv"

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "locations_path": "data/english_locations.csv",
        "app_title": "🏖️ Travel Semantic Search",
        "app_description": "Search for tourist attractions, events, and news in Thailand using AI",
        "search_placeholder": "Type keywords like 'beautiful beach' or 'cultural festival'",
        "results_header": "Search Results",
        "no_results": "No results found for your search",
        "relevance": "Relevance",
        "province": "Province",
        "location": "Location",
        "date": "Date",
        "type": "Type",
        "author": "Author"
    },
    "th": {
        "name": "ไทย",
        "locations_path": "data/thai_locations.csv",
        "app_title": "🏖️ ค้นหาแหล่งท่องเที่ยว",
        "app_description": "ค้นหาสถานที่ท่องเที่ยว กิจกรรม และข่าวสารในประเทศไทยด้วย AI",
        "search_placeholder": "พิมพ์คำค้นหา เช่น 'ทะเลสวย น้ำใส' หรือ 'เทศกาลประเพณี'",
        "results_header": "ผลการค้นหา",
        "no_results": "ไม่พบผลการค้นหาที่ตรงกับคำค้น",
        "relevance": "ความเกี่ยวข้อง",
        "province": "จังหวัด",
        "location": "สถานที่",
        "date": "วันที่",
        "type": "ประเภท",
        "author": "ผู้เขียน"
    }
}

# Application settings
DEFAULT_LANGUAGE = "th"

# UI settings
PAGE_ICON = "🏖️"
LAYOUT = "wide"
