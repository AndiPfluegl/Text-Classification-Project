# Text-Classification-Project
This project involves the development of a text classification system using machine learning. The system is trained on two datasets, the 20 Newsgroups and Reuters, to classify text into predefined categories.

## Files in the Project
***
Training.py: Contains the code for data preprocessing, model training, and saving the trained model and vectorizer.
main.py: Used for loading the trained model and vectorizer, preprocessing new text data, and predicting the category.

## Getting Started
***

### Prerequisites
***
* Python 3
* Libraries: scikit-learn, NLTK, joblib

### Installation
***
Install the required Python packages:
$ pip install scikit-learn nltk joblib

Run Training.py to train the model. This script will save the trained model and vectorizer to disk.
Use main.py to predict the category of new text data.

## Usage
***
To use the text classification system:
Place your text file (e.g., input.txt) in the project directory.
Run main.py. The script will output the predicted category of the text in input.txt.

## How It Works
***
Training.py uses TF-IDF for feature extraction and trains two models: Multinomial Naive Bayes and SVM. The trained models along with the TF-IDF vectorizer are saved for later use.
main.py loads the trained model and vectorizer, preprocesses the input text, and uses the model to predict the text's category.

## Contributing
***
Feel free to fork this project and submit pull requests for improvements or bug fixes.
