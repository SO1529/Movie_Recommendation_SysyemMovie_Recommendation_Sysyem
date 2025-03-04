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

