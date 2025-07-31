# 🎬 Movie Recommendation System
A sophisticated **AI-powered movie recommendation system** built with Flask, Python, and Machine Learning. This system uses content-based filtering
## 🌟 Features
- **🤖 AI-Powered Recommendations**: Uses machine learning algorithms for content-based filtering
- **🎯 Smart Search**: Auto-complete movie search with real-time suggestions
- **🎨 Beautiful UI**: Modern, responsive web interface with smooth animations
- **⚡ Fast Performance**: Optimized similarity calculations using cosine similarity

  ## 🚀 Quick Start

  ### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/furkhan67/movie_recommender.git
   cd movie_recommender
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv movie_env
   source movie_env/bin/activate  # On Windows: movie_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (first time only)
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. **Prepare your data**
   - Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the `data/` folder
   - Run the Jupyter notebook `analysis.ipynb` to generate `model.pkl` and `similarity.pkl`

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Open your browser** and navigate to `http://localhost:5000`

## 📊 Data Analysis Workflow

### Step 1: Run the Jupyter Notebook

Open `analysis.ipynb` and run all cells sequentially. The notebook will:

1. **Load and explore** the movie datasets
2. **Clean and preprocess** the data
3. **Extract features** from genres, cast, crew, and keywords
4. **Create content tags** combining all features
5. **Apply text preprocessing** with stemming and tokenization
6. **Generate feature vectors** using CountVectorizer
7. **Calculate similarity matrix** using cosine similarity
8. **Save the model** as pickle files

### Key Notebook Sections:

- **Data Loading**: Import and merge movie and credits datasets
- **Feature Engineering**: Extract meaningful features from JSON-like strings
- **Text Processing**: Advanced NLP preprocessing with NLTK
- **Model Training**: Create a similarity matrix using machine learning
- **Model Evaluation**: Test recommendations with sample movies
- **Model Persistence**: Save trained model for Flask app

## 🛠️ How It Works

### Content-Based Filtering Algorithm

1. **Feature Extraction**: Extract and combine multiple movie features:
   - **Genres**: Action, Comedy, Drama, etc.
   - **Cast**: Top 3 main actors
   - **Crew**: Director information
   - **Keywords**: Plot-related keywords
   - **Overview**: Movie plot summary

2. **Text Preprocessing**: 
   - Convert to lowercase
   - Remove punctuation
   - Apply stemming using Porter Stemmer
   - Remove stop words

3. **Vectorisation**: Convert text features into numerical vectors using CountVectorizer

4. **Similarity Calculation**: Use cosine similarity to find movies with similar content

5. **Recommendation Generation**: Return top 5 most similar movies

### Mathematical Foundation

The system uses **cosine similarity** to measure the similarity between movies:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Where A and B are feature vectors representing movies.

## 📁 Project Structure

```
movie-recommender/
├── analysis.ipynb          # Jupyter notebook for model training
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── model.pkl              # Trained model (generated)
├── similarity.pkl         # Similarity matrix (generated)
├── templates/
│   ├── index.html         # Home page template
│   └── recommend.html     # Recommendations page template
└── data/
    ├── tmdb_5000_movies.csv    # Movie dataset
    └── tmdb_5000_credits.csv   # Credits dataset
```
##  📊 Dataset Information

The system uses the **TMDb 5000 Movie Dataset** which includes:

- **4,803 movies** with detailed metadata
- **Movie features**: Genres, keywords, cast, crew, overview, ratings
- **Time period**: Movies from various decades
- **Data format**: CSV files with JSON-encoded feature columns




