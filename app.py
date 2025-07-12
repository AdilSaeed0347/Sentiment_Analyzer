from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
from pathlib import Path
import logging
import socket
try:
    import emoji
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'emoji' library is required. Please install it using: pip install emoji")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Keyword rules for fallback
KEYWORD_RULES = {
    'Risky': [
        'kill', 'murder', 'die', 'attack', 'hurt', 'harm', 'destroy', 'ruin', 'threaten', 'stab',
        'shoot', 'bomb', 'assault', 'violence', 'beat', 'slaughter', 'end you', 'get you',
        'mess you up', 'done for', 'fuck you', 'take you down', 'wipe out', 'regret',
        'watch your back', 'youre dead', 'pay for this', 'gonna die', 'make you suffer',
        'slap you', 'hunt you down', 'fuck u up', 'final warning', 'prepare to suffer',
        'youll regret this', 'find you', 'no escape', 'last chance', 'cross me', 'smoke you',
        'wreck you', 'get rekt', 'youre finished', 'better run', '😡', '😈', '⚠️', '🔪',
        '💀', '🧨', '🚨', '☠️', '🪓', '🔫', '⚡', '🩸', '💥'
    ],
    'Negative': [
        'trash', 'garbage', 'shitty', 'wack', 'lame', 'awful', 'terrible', 'horrible', 'bad',
        'suck', 'mess', 'fucked up', 'bullshit', 'crap', 'pathetic', 'stupid', 'dumb', 'useless',
        'worst', 'screwed up', 'messed up', 'no good', 'waste of time', 'smh', 'ngl', 'total fail',
        'liar', 'lier', 'illiterate', 'idiot', 'moron', 'cringe', 'embarrassing', 'pathetic',
        'failure', 'loser', 'joke', 'clown', 'bozo', 'trashy', 'disgusting', 'gross', 'nonsense',
        'pointless', 'waste', 'not accept', 'dont accept', 'reject', 'nope', 'shit', 'damn',
        'whack', 'bunk', 'rubbish', 'stinks', 'lousy', '🙅', '😒', '😖', '😞', '😩', '💔',
        '🤦', '🙄', '😣', '😤', '🤢', '🤮', '😢', '😠'
    ],
    'Sarcastic': [
        'genius', 'brilliant', 'amazing', 'fantastic', 'perfect', 'epic', 'stellar', 'legendary',
        '*great*', '*so*', '…not', 'nice one', 'good going', 'totally gonna', '*impressed*',
        'youre a pro', 'such a vibe', 'sure', 'whatever', 'cool story', 'nice try', 'yup',
        'definitely', 'obviously', 'sure thing', 'big brain', 'masterpiece', 'game changer',
        'top tier', 'nailed it', 'slow clap', 'bravo', 'well done', 'pro move', 'einstein',
        'genius move', 'so smart', 'wowza', '🙄', '😏', '🤦', '🙃', '😉', '😂', '😜', '😆'
    ],
    'Positive': [
        'awesome', 'great', 'dope', 'fire', 'lit', 'amazing', 'fantastic', 'epic', 'sick',
        'stellar', 'cool', 'bomb', 'slay', 'crush', 'kill it', 'nailed', 'smashed', 'love',
        'vibe', 'energy', 'banger', 'iconic', 'perfect', 'keep it up', 'good job', 'well done',
        'congratulations', 'congraulations', 'proud', 'good boy', 'good girl', 'pokie', 'cutie',
        'champ', 'winner', 'hero', 'legend', 'goat', 'slaps', 'bussin', 'vibes', 'on point',
        'nailed it', 'killed it', 'you rock', 'so proud', 'amaze', 'fabulous', 'yay', '🔥',
        '😊', '👍', '😄', '💯', '👏', '🥳', '🙌', '💖', '🎉', '🌟', '💪', '😍', '✨', '😎'
    ],
    'Neutral': [
        'fine', 'okay', 'cool', 'alright', 'chill', 'decent', 'meh', 'normal', 'standard',
        'whatever', 'I get it', 'seems fine', 'no big deal', 'just saying', 'ig', 'tbh',
        'no shade', 'behave', 'fair', 'not bad', 'so so', 'kinda', 'eh', 'all good',
        'no comment', 'its fine', 'no issue', 'just okay', 'right way', 'makes sense',
        'seems legit', '😐', '🤷', '👍', '👌', '😶', '😑', '🤔', '🙂'
    ],
    'Invalid': [
        'asdjkl', 'qwerty', '!!!', '@#$%^', 'lkjhgfd', '123456', '', 'zxcvb', '???', '!!!???',
        'punctuation', 'randomtext', 'kjhgf', 'qwert', 'asdf', 'zxcvbn', '09876', '!!!!',
        '**##@@', 'a', '1', '@', '###', '+++=', 'xyxyxy', 'qazwsx', 'plmokn', 'jklmno',
        'randomz', 'wtf123', 'xxx', 'jjj', 'kkk', '###!!!'
    ]
}

DUAL_MEANING_RULES = {
    'fire': {'Positive': ['awesome', 'lit', 'dope', 'great'], 'Risky': ['burn', 'destroy'], 'Negative': ['bad', 'awful']},
    'sick': {'Positive': ['cool', 'dope', 'awesome'], 'Negative': ['gross', 'bad', 'ill']},
    'bomb': {'Positive': ['awesome', 'lit', 'great'], 'Risky': ['explode', 'attack'], 'Negative': ['fail', 'terrible']},
    'kill': {'Positive': ['slay', 'crush', 'nailed', 'killed it'], 'Risky': ['murder', 'die', 'hurt'], 'Negative': ['bad', 'awful']},
    '🔥': {'Positive': ['awesome', 'lit', 'dope', 'great'], 'Risky': ['burn', 'destroy'], 'Negative': ['bad', 'awful']},
    'bad': {'Positive': ['cool', 'dope', 'sick'], 'Negative': ['awful', 'terrible', 'suck'], 'Sarcastic': ['joke', 'pathetic']},
    'lit': {'Positive': ['awesome', 'fire', 'dope'], 'Negative': ['mess', 'trash'], 'Sarcastic': ['sure', 'whatever']}
}

def check_keywords(text, label):
    text_lower = text.lower()
    text_demojized = emoji.demojize(text_lower)
    keywords = KEYWORD_RULES.get(label, [])
    keyword_pattern = '|'.join(r'\b{}\b'.format(re.escape(k)) for k in keywords if len(k) > 1)
    if re.search(keyword_pattern, text_demojized) or any(k in text for k in keywords):
        return True
    if label == 'Risky' and re.search(r'\b(will|gonna|going to|better)\b.*\b(kill|die|hurt|destroy|ruin|fuck you|regret|pay)\b', text_demojized):
        return True
    if label == 'Negative' and re.search(r'\b(fuck|shit|damn)\b(?! you| up)', text_demojized):
        return True
    if label == 'Negative' and re.search(r'\b(why are you so|you are a|youre a)\b.*\b(stupid|dumb|idiot|moron|illiterate|lier|liar|useless|pathetic|waste|loser)\b', text_demojized):
        return True
    if label == 'Sarcastic' and re.search(r'\b(\.\.\.|sure|whatever|cool story|nice try|yup|definitely|obviously)\b', text_demojized):
        return True
    for keyword, rules in DUAL_MEANING_RULES.items():
        if (keyword in text or keyword in text_demojized) and label in rules:
            context_words = rules[label]
            context_pattern = '|'.join(r'\b{}\b'.format(re.escape(w)) for w in context_words)
            if re.search(context_pattern, text_demojized):
                return True
    return False

def is_invalid(text):
    text = text.strip()
    if emoji.distinct_emoji_list(text) and not re.sub(r'[^\w\s]', '', text).strip():
        return False
    if len(text) < 3 or re.match(r'^[!@#$%^&*()+=?]+$', text) or text == '':
        return True
    if re.match(r'^[a-z0-9]{3,6}$', text) and not any(word in text.lower() for word in ['you', 'are', 'is', 'this']):
        return True
    return False

# Load models
def load_models():
    models_dir = Path('models')
    required_files = ['sentiment_model.pkl', 'tfidf_vectorizer.pkl', 'label_encoder.pkl']
    for file in required_files:
        if not (models_dir / file).exists():
            raise FileNotFoundError(f"Missing model file: {file}")
    try:
        model = joblib.load(models_dir / 'sentiment_model.pkl')
        vectorizer = joblib.load(models_dir / 'tfidf_vectorizer.pkl')
        label_encoder = joblib.load(models_dir / 'label_encoder.pkl')
        assert hasattr(model, 'predict_proba'), "Model missing predict_proba"
        assert hasattr(vectorizer, 'vocabulary_'), "Invalid vectorizer"
        assert hasattr(label_encoder, 'classes_'), "Invalid label encoder"
        return model, vectorizer, label_encoder
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise

try:
    model, vectorizer, label_encoder = load_models()
    logger.info(" Models loaded successfully")
except Exception as e:
    logger.critical(f"❌ Failed to load models: {str(e)}")
    exit(1)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'comment' not in data:
            return jsonify({'error': 'Missing comment field'}), 400
        comment = data['comment'].strip()
        if not comment:
            return jsonify({'error': 'Empty comment'}), 400

        # Check for invalid input
        if is_invalid(comment):
            return jsonify({
                'prediction': 'Invalid',
                'confidence': 1.0,
                'error': None
            })

        # Check keywords for sentiment
        for label in ['Risky', 'Negative', 'Sarcastic', 'Positive', 'Neutral']:
            if check_keywords(comment, label):
                return jsonify({
                    'prediction': label,
                    'confidence': 1.0,
                    'error': None
                })

        # Fallback to ML model
        cleaned_text = clean_text(comment)
        vader_score = analyzer.polarity_scores(emoji.emojize(cleaned_text))['compound']
        X_new = vectorizer.transform([cleaned_text]).toarray()
        X_new = np.hstack((X_new, [[vader_score]]))
        probs = model.predict_proba(X_new)[0]
        max_prob = max(probs)
        pred = model.predict(X_new)
        predicted_label = label_encoder.inverse_transform(pred)[0]

        # Confidence thresholds
        CONFIDENCE_THRESHOLDS = {
            'Risky': 0.7,
            'Negative': 0.65,
            'Sarcastic': 0.6,
            'Positive': 0.55,
            'Neutral': 0.5,
            'Invalid': 0.5
        }

        if max_prob < CONFIDENCE_THRESHOLDS.get(predicted_label, 0.5):
            predicted_label = 'Ambiguous'

        return jsonify({
            'prediction': predicted_label,
            'confidence': float(max_prob),
            'error': None
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Server initialization
def find_available_port(start_port):
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1

if __name__ == '__main__':
    port = find_available_port(5000)
    if port != 5000:
        logger.warning(f"Port 5000 in use, using port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
