# Travel Semantic Search

A multilingual semantic search application for tourist attractions, events, and news in Thailand using AI.

## Project Structure

```
semantic-search/
├── src/                    # Source code
│   ├── __init__.py
│   ├── app.py             # Main Streamlit application
│   └── config.py          # Configuration settings
├── data/                  # Data files
│   ├── articles.csv
│   ├── english_locations.csv
│   ├── events.csv
│   └── thai_locations.csv
├── tests/                 # Test files
│   └── __init__.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Features

- Multilingual support (Thai and English)
- Semantic search using AI embeddings
- Search across multiple data types:
  - Tourist locations
  - Events
  - News articles

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
streamlit run src/app.py
```

## Technology Stack

- Python
- Streamlit
- Sentence Transformers
- scikit-learn
- pandas
- langdetect
