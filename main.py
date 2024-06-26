# main.py

from flask import Flask, render_template, request, jsonify
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import os

app = Flask(__name__)

# Load the model and TF-IDF vectorizer
models_dir = 'models'
model_filepath = os.path.join(models_dir, 'logreg_model.pkl')
vectorizer_filepath = os.path.join(models_dir, 'tfidf_vectorizer.pkl')

with open(model_filepath, 'rb') as model_file:
    logreg_model = pickle.load(model_file)

with open(vectorizer_filepath, 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha() and word not in stop_words]
        return ' '.join(tokens)
    else:
        return ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.get_json()['text']
        processed_text = preprocess_text(text)
        vectorized_text = tfidf_vectorizer.transform([processed_text])
        prediction = logreg_model.predict(vectorized_text)[0]

        return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
