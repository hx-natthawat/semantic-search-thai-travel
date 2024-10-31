"""
Configuration settings for the Thai Travel Semantic Search application.
"""

# Model settings
MODEL_NAME = "jinaai/jina-embeddings-v3"
TOP_K_RESULTS = 3

# Data settings
DATA_PATH = "data/thai_locations.csv"

# Application settings
APP_TITLE = "🏖️ Thai Travel Semantic Search"
APP_DESCRIPTION = "ค้นหาสถานที่ท่องเที่ยวในประเทศไทยด้วย AI"
SEARCH_PLACEHOLDER = "พิมพ์คำค้นหา เช่น 'ทะเลสวย น้ำใส' หรือ 'ประวัติศาสตร์'"

# UI settings
PAGE_ICON = "🏖️"
LAYOUT = "wide"
