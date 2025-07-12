from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import logging
import socket

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Load models with version check
try:
    import sklearn
    assert sklearn.__version__ == '1.6.1', f"Requires scikit-learn 1.6.1, found {sklearn.__version__}"
    
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    logging.info("✅ Models loaded successfully")
except Exception as e:
    logging.error(f"❌ Model loading failed: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        comment = data.get('comment', '').strip()
        
        if not comment:
            return jsonify({'error': 'No comment provided'}), 400

        features = vectorizer.transform([comment])
        if features.shape[1] < model.n_features_in_:
            padding = csr_matrix((1, model.n_features_in_ - features.shape[1]))
            features = hstack([features, padding])
        
        prediction = label_encoder.inverse_transform(model.predict(features))[0]
        return jsonify({'prediction': prediction, 'error': None})

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = 5000
    if is_port_in_use(port):
        port = 5001
        logging.warning(f"Port 5000 in use, falling back to {port}")
    
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)