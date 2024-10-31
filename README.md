# Thai Travel Semantic Search

A semantic search application for Thai tourist locations using Streamlit and the jinaai/jina-embeddings-v3 model.

## Features

- Semantic search for Thai tourist locations
- Real-time search results with similarity scores
- Thai language support
- Interactive web interface

## Installation

1. Clone the repository:

```bash
git clone [your-repo-url]
cd semantic-search
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:

```bash
streamlit run app.py
```

2. Open your browser and navigate to http://localhost:8501

3. Enter your search query in Thai, for example:
   - "ทะเลสวย น้ำใส"
   - "ประวัติศาสตร์"
   - "วัดเก่าแก่"

## Project Structure

```
semantic-search/
├── app.py              # Main Streamlit application
├── config.py           # Configuration settings
├── data/
│   └── thai_locations.csv  # Sample dataset
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Data

The application uses a curated dataset of Thai tourist locations including:

- Location names
- Detailed descriptions
- Provinces

## Model

Uses the `jinaai/jina-embeddings-v3` model for generating embeddings, which has good support for Thai language.
