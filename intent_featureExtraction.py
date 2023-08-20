from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

# Load preprocessed data
data = pd.read_csv('preprocessed_intent_data.csv')

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed sentences
tfidf_matrix = tfidf_vectorizer.fit_transform(data['preprocessed_sentence'])

# Save TF-IDF matrix
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')