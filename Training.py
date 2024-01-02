from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from joblib import dump
import nltk
from nltk.corpus import reuters
from collections import Counter
from sklearn.svm import SVC

def preprocess_text(text):
    '''takes an input string and removes special characters and numbers, convert to lowercase, performs tokenization and
    lemmatizion and returns the text as a string'''
    #Remove special characters and numbers from the text
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


# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers', 'footers', 'quotes'))

# Load the Reuters corpus
nltk.download('reuters')
nltk.download('punkt')

# Extract the data and labels from 20 Newsgroups
texts = newsgroups.data
labels = newsgroups.target

# Count the examples in each category for 20 Newsgroups
newsgroups_counts = Counter(labels)

# Preprocess the texts from 20 Newsgroups
processed_texts = [preprocess_text(text) for text in texts]

# Filter categories with less than 100 examples in 20 Newsgroups
filtered_newsgroups_indices = [i for i, label in enumerate(labels) if newsgroups_counts[label] >= 100]
filtered_newsgroups_texts = [processed_texts[i] for i in filtered_newsgroups_indices]
filtered_newsgroups_labels = [labels[i] for i in filtered_newsgroups_indices]

# Preprocess the Reuters Corpus
reuters_texts = [' '.join(reuters.words(fileid)) for fileid in reuters.fileids()]
reuters_labels = [reuters.categories(fileid)[0] for fileid in reuters.fileids()]
processed_reuters_texts = [preprocess_text(text) for text in reuters_texts]

# Create a mapping for Reuters labels
unique_reuters_labels = set(reuters_labels)
label_mapping = {label: i + 20 for i, label in enumerate(unique_reuters_labels)}

# Convert Reuters labels into numeric values
numerical_reuters_labels = [label_mapping[label] for label in reuters_labels]

# Count the examples in each category for Reuters
reuters_counts = Counter(reuters_labels)

# Filter categories with less than 100 examples in Reuters
filtered_reuters_indices = [i for i, label in enumerate(reuters_labels) if reuters_counts[label] >= 100]
filtered_reuters_texts = [processed_reuters_texts[i] for i in filtered_reuters_indices]
filtered_reuters_labels = [numerical_reuters_labels[i] for i in filtered_reuters_indices]

# Combine the filtered texts and labels from both datasets
combined_texts = filtered_newsgroups_texts + filtered_reuters_texts
combined_labels = filtered_newsgroups_labels + filtered_reuters_labels

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X_combined = tfidf.fit_transform(combined_texts)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, combined_labels, test_size=0.2, random_state=42)

# Train the model using the Multinomial Naive Bayes algorithm
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set and evaluate the model
y_pred = model.predict(X_test)
print('Naive Bayes: ')
print(classification_report(y_test, y_pred))

# Create and train the model with SVM
model = SVC()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print('SVM: ')
print(classification_report(y_test, y_pred))

# Save the trained model
dump(model, 'trained_model.joblib')

# Save the TF-IDF vectorizer
dump(tfidf, 'tfidf_vectorizer.joblib')

# Save the category names for the datasets
# Original category names from the 20 Newsgroups dataset
newsgroups_category_names = newsgroups.target_names
# Add Reuters category names, sorted by their numerical values
reuters_category_names = [label for label, index in sorted(label_mapping.items(), key=lambda x: x[1])]

# Combine the categories of the two datasets and save it with joblib
combined_category_names = newsgroups_category_names + reuters_category_names
dump(combined_category_names, 'category_names.joblib')