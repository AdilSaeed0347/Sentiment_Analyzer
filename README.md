Sentiment Analysis for Social Media
A Flask app to classify social media comments into Positive, Negative, Sarcastic, Risky, Neutral, and Invalid categories using a RandomForestClassifier with TF-IDF, VADER, and keyword rules.
Features

Classifies comments with slang, typos, and emojis.
Provides a web interface and API for predictions.

Requirements

Python 3.11
Libraries: flask, joblib, emoji, vaderSentiment, pandas, scikit-learn, numpy

Setup

Clone the repository.
Install dependencies: pip install -r requirements.txt
Create models directory.

Usage

Train the model: python train.py
Run the app: python app.py
Access at http://localhost:5000 or use the API.

Project Structure

train.py: Trains the model.
app.py: Flask app for predictions.
templates/index.html: Web interface.
models/: Stores model artifacts.
requirements.txt: Lists dependencies.

Example Predictions

"you are good boy" → Positive
"kill you bab" → Risky
"asdjkl" → Invalid
