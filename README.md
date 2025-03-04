# Movie Recommendation System

## Overview
This project implements a Movie Recommendation System that suggests movies based on user preferences using machine learning techniques. The system leverages collaborative filtering, content-based filtering, and hybrid approaches to generate recommendations.


## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure that `requirements.txt` contains the following dependencies:

```
numpy
pandas
scikit-learn
nltk
tensorflow
flask (if deploying as a web app)
```

## Usage

1. **Load the Dataset**:
   - The system uses movie rating datasets like **MovieLens** or **IMDB datasets**.
   - Load the dataset using Pandas:
   ```python
   import pandas as pd
   movies = pd.read_csv('movies.csv')
   ratings = pd.read_csv('ratings.csv')
   ```

2. **Preprocessing**:
   - Clean and format the dataset.
   - Tokenize and vectorize movie descriptions (for content-based filtering).

3. **Building the Model**:
   - Implement **Collaborative Filtering** (User-User & Item-Item)
   - Implement **Content-Based Filtering** using **TF-IDF Vectorization**.
   - Hybrid approach (combining both techniques for better accuracy).

4. **Make Recommendations**:
   - Generate movie recommendations based on a user’s past preferences.
   - Display the top-N recommended movies.

## Features

- **Content-Based Filtering**: Recommends movies based on similarity in genre, description, and features.
- **Collaborative Filtering**: Uses past user ratings to recommend movies.
- **Hybrid Approach**: Combines multiple recommendation techniques for improved accuracy.
- **Flask API Deployment**: Optionally deploy the system as a web application.

## Dependencies

- `numpy` - For numerical operations.
- `pandas` - For data manipulation.
- `scikit-learn` - For machine learning models.
- `nltk` - For natural language processing (text-based recommendations).
- `tensorflow` - For deep learning-based recommendations.
- `flask` - To deploy as a web app (optional).

## Model Overview

### 1. Content-Based Filtering:
   - Uses **TF-IDF Vectorization** to compare movie descriptions.
   - Finds similarity using **cosine similarity**.
   
### 2. Collaborative Filtering:
   - Uses **User-User** and **Item-Item** similarity.
   - Implements **Matrix Factorization** (SVD or deep learning models).
   
### 3. Hybrid Model:
   - Combines both filtering methods for better performance.

### TF-IDF for Content-Based Filtering:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english')
movie_matrix = vectorizer.fit_transform(movies['description'])
similarity = cosine_similarity(movie_matrix)
```

## Example Usage

### Generating Recommendations for a User
```python
def recommend_movie(movie_title, num_recommendations=5):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]
```


