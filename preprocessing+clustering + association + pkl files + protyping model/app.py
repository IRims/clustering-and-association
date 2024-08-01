import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Load the models
with open('kmeans_clustering_model.pkl', 'rb') as file:
    kmeans = pickle.load(file)

with open('pca_model.pkl', 'rb') as file:
    pca = pickle.load(file)

with open('association_rules.pkl', 'rb') as file:
    rules = pickle.load(file)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000)

# Function to preprocess text
def preprocess_text(text):
    import re
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\W', ' ', text)   # Remove special characters
    text = text.lower()               # Convert to lowercase
    tokens = text.split()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Streamlit UI
st.title("Real-Time Bullying Statement Detection")

# Input text
input_text = st.text_area("Enter a statement:", "")

if st.button("Analyze"):
    if input_text:
        processed_text = preprocess_text(input_text)
        tfidf_matrix = vectorizer.fit_transform([processed_text])

        # Dimensionality reduction
        reduced_data = pca.transform(tfidf_matrix.toarray())

        # Clustering
        cluster = kmeans.predict(reduced_data)[0]

        st.write(f"The statement is categorized into cluster: {cluster}")

        # Visualization
        pca_df = pd.DataFrame(pca.transform(vectorizer.fit_transform(df['tweet_text']).toarray()), columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = kmeans.predict(pca_df)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', alpha=0.6)
        plt.scatter(reduced_data[0, 0], reduced_data[0, 1], c='red', s=100)  # Highlight the new input statement
        st.pyplot(plt)

        # Association Rules
        st.write("Association Rules:")
        st.write(rules)
    else:
        st.error("Please enter a statement to analyze.")

