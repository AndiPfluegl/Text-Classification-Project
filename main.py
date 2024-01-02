from joblib import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

#Load the trained model via joblib
model = load('trained_model.joblib')

# oad the TF-IDF vectorizer via joblib
tfidf_vectorizer = load('tfidf_vectorizer.joblib')


def preprocess_text(text):
    '''takes an input string and removes special characters and numbers, convert to lowercase, performs tokenization and
    lemmatizion and returns the text as a string'''
    # Remove special characters and numbers from the text
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

    # Convert text to lowercase for uniformity
    text = text.lower()

    # Tokenization: Splitting the text into individual words
    tokens = text.split()

    # Remove stopwords and perform lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Return the preprocessed text as a single string
    return ' '.join(tokens)

def predict_category(file_path, tfidf_vectorizer, model):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Convert the text into TF-IDF features
    text_tfidf = tfidf_vectorizer.transform([processed_text])

    # Load the category names
    category_names = load('category_names.joblib')

    # Predict the category
    predicted_category_index = model.predict(text_tfidf)[0]

    # Get the predicted category name from its index
    predicted_category = category_names[predicted_category_index]

    return predicted_category

# Path to the text file
file_path = 'input.txt'

# Predict the category of the text and output it on the prompt
category = predict_category(file_path, tfidf_vectorizer, model)
print(f"Predicted Category: {category}")