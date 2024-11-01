"""
Configuration settings for the Thai Travel Semantic Search application.
"""

# Model settings
MODEL_NAME = "jinaai/jina-embeddings-v3"
TOP_K_RESULTS = 3

# Data settings
LOCATIONS_PATH = "data/thai_locations.csv"
EVENTS_PATH = "data/events.csv"
ARTICLES_PATH = "data/articles.csv"

# Application settings
APP_TITLE = "🏖️ Thai Travel Semantic Search"
APP_DESCRIPTION = "ค้นหาสถานที่ท่องเที่ยว กิจกรรม และข่าวสารในประเทศไทยด้วย AI"
SEARCH_PLACEHOLDER = "พิมพ์คำค้นหา เช่น 'ทะเลสวย น้ำใส' หรือ 'เทศกาลประเพณี'"

# UI settings
PAGE_ICON = "🏖️"
LAYOUT = "wide"
