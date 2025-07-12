import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
import joblib
import logging
try:
    import emoji
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'emoji' library is required. Please install it using: pip install emoji")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Expanded keyword rules with more slang, typos, and emojis
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

def generate_examples(base_examples, templates, replacements, target_count=5000):
    examples = base_examples.copy()
    while len(examples) < target_count:
        template = random.choice(templates)
        for placeholder, options in replacements.items():
            template = template.replace(placeholder, random.choice(options))
        examples.append(template)
    return examples[:target_count]

# Positive examples
positive_templates = [
    "Yo {sub}, this is {adjective} {emoji}!", "Ur {noun} is {adjective} 🔥",
    "Love how u {verb} this tbh", "This {noun} slaps {emoji}!",
    "You’re {adverb} {adjective}, keep it up!", "What a {adjective} {noun} 😍",
    "{exclamation}, ur {noun} is {adjective}!", "This is so {adjective} fr 💯"
]
positive_replacements = {
    '{adjective}': KEYWORD_RULES['Positive'][:20],
    '{noun}': ['vibe', 'post', 'work', 'idea', 'content', 'style', 'move'],
    '{verb}': ['nailed', 'killed', 'crushed', 'smashed', 'rocked', 'slayed'],
    '{adverb}': ['so', 'hella', 'super', 'totally', 'mad', 'insanely'],
    '{sub}': ['bro', 'dude', 'fam', 'mate', 'homie', 'queen', 'king', 'champ'],
    '{exclamation}': ['Yo', 'Wow', 'Damn', 'Hell yeah', 'Ayy'],
    '{emoji}': ['🔥', '😍', '💯', '👏', '🚀', '😊', '👍', '🥳', '🙌', '💖', '🎉', '🌟', '💪']
}
positive_base = [
    "Yo this is straight fire 🔥", "Ur post is so dope fam! 💯", "Love how you crushed this tbh 😍",
    "Bro you’re killing it fr! 🚀", "This vibe is hella awesome 😊", "Ayy, this content slaps! 👏",
    "You nailed this post, so lit! 💖", "Damn, what a sick idea 🥳", "This is epic, keep it up! 👍",
    "Ur work is absolutely fire, homie! 🔥", "Good job, you slayed this 💯",
    "This is fantastic, no cap! 😄", "You’re a vibe, keep it real 🔥", "you are good boy 😊",
    "congraulations on ur win! 🎉", "ur so dope fr 💪", "this post is a banger! 🔥",
    "how are you pokie 😄", "youre a legend mate! 👏", "youre a champ! 🌟",
    "this is so dope fr 💖", "keep shining king! ✨", "you rock bro! 😎"
]

# Negative examples
negative_templates = [
    "This is {adjective} af 😒", "Why u {verb} this {noun}?", "Ur {noun} is straight {adjective}",
    "This {noun} is {adverb} {adjective} smh", "Bro, you’re {verb} this {adjective} 😖",
    "What a {adjective} {noun} fr", "This is just {adjective} {emoji}",
    "Ur {adjective} {noun} is a no for me", "Why are you so {adjective}?"
]
negative_replacements = {
    '{adjective}': KEYWORD_RULES['Negative'][:20],
    '{noun}': ['post', 'content', 'idea', 'take', 'work', 'vibe'],
    '{verb}': ['screwed', 'messed', 'fucked', 'botched', 'ruined'],
    '{adverb}': ['so', 'hella', 'super', 'totally', 'straight up'],
    '{emoji}': ['😒', '😖', '🤦', '🙅', '😞', '💔', '😩', '🤢', '🤮']
}
negative_base = [
    "This is straight trash 😒", "Why you posting this garbage ngl? 😖", "Ur idea is so wack smh 🤦",
    "Bro this content is hella shitty 😞", "You fucked this up fr 🙅", "What a lame post tbh 😩",
    "This vibe is mad terrible 💔", "You botched this so bad 😞", "This is just sus af 🤦",
    "Ur take is pure garbage, no cap 😒", "This is fucked up ngl 😑",
    "you are a lier 😞", "why are you so illiterate 🤦", "i not accept what are you saying 😒",
    "you are waste of my time 💔", "this is total crap smh 😩", "ur so dumb fr 🤦",
    "why u gotta be so stupid? 😒", "this post is embarrassing 😖", "youre such a loser 😩",
    "this is pointless ngl 🤮", "why are you so moron? 😒", "ur take is trashy fr 💔"
]

# Sarcastic examples
sarcastic_templates = [
    "Oh *{adjective}* {noun}, {sub}…", "Wow, you’re *so* {adjective} {emoji}",
    "Sure, this is {adverb} {adjective} 🙄", "What a *{adjective}* {noun}…not",
    "Bro, you {verb} this *{adjective}* fr", "I’m *{adverb}* impressed…",
    "{exclamation}, such {adjective} work 😏", "Yeah, *{adjective}* move {sub}…"
]
sarcastic_replacements = {
    '{adjective}': KEYWORD_RULES['Sarcastic'][:12],
    '{noun}': ['move', 'post', 'idea', 'vibe', 'content', 'work'],
    '{verb}': ['nailed', 'killed', 'slayed', 'crushed'],
    '{adverb}': ['so', 'hella', 'super', 'totally', 'mad'],
    '{sub}': ['bro', 'dude', 'genius', 'pro', 'legend'],
    '{exclamation}': ['Wow', 'Ayy', 'Sure', 'Nice'],
    '{emoji}': ['🙄', '😏', '😉', '🤦', '🙃', '😂']
}
sarcastic_base = [
    "Oh *genius* move, bro… 🙄", "Wow, you’re *so* brilliant 😏", "Sure, this is *totally* fire…not 🙃",
    "What a *legendary* post…not 🤦", "I’m *hella* impressed fr… 😏", "Ayy, such *epic* content 😉",
    "You crushed this *so* well…not 🙄", "Yeah, *amazing* idea dude 😏", "This is *so* gonna slap… 🙃",
    "Ur *perfect* vibe is iconic…not 🤦", "Nice one, *genius* 🙃",
    "You clearly don’t know what you're doing 🤦", "Wow, such a *pro* move… 🙄",
    "Sure thing, legend 😏", "This is *definitely* a banger…not 🙃", "Big brain energy fr 😂"
]

# Risky examples
risky_templates = [
    "You’ll {verb} for this {emoji}", "I’m gonna {verb} you {sub}…", "You’re {verb} fr",
    "This’ll {verb} you bad {emoji}", "Better {verb} yourself {sub}", "I’ll {verb} you ngl",
    "You’re in for some {adjective} {noun}…", "Don’t make me {verb} you {emoji}",
    "You won’t {verb} this {sub}", "This is ur last chance to {verb} 😡"
]
risky_replacements = {
    '{verb}': KEYWORD_RULES['Risky'][:12],
    '{noun}': ['trouble', 'pain', 'hell', 'danger', 'heat'],
    '{adjective}': ['bad', 'deep', 'serious', 'real', 'messed up'],
    '{sub}': ['bro', 'dude', 'fam', 'mate', 'homie'],
    '{emoji}': ['😡', '😈', '⚠️', '🔪', '💀', '🧨', '🚨', '☠️', '🪓']
}
risky_base = [
    "I will kill you 😡", "You’re gonna die fr 💀", "I’ll fuck you up bro ⚠️",
    "You better watch your back… 😈", "You’re done for, mate 😡", "I’ll make you regret this fr 🔪",
    "You’re dead if you try that again 💀", "I’ll mess you up bad… 😈",
    "This’ll ruin your career ngl 🔪", "I’ll get you, no cap 😡",
    "fuck you bab 😡", "This will ruin you fr 💀", "watch your back homie ⚠️",
    "youre done for 😈", "dont test me 😡", "slap you silly 🪓", "hunt you down 😈"
]

# Neutral examples
neutral_templates = [
    "This is {adjective} ig", "Ur {noun} seems {adjective}", "Thanks for {verb} this {emoji}",
    "Looks {adjective} to me", "I get ur {noun} tbh", "This {noun} is {adjective} fr",
    "Seems {adjective} ngl", "Just a {adjective} {noun}", "I {verb} what u mean"
]
neutral_replacements = {
    '{adjective}': KEYWORD_RULES['Neutral'][:10],
    '{noun}': ['post', 'take', 'idea', 'content', 'point'],
    '{verb}': ['sharing', 'posting', 'saying', 'dropping'],
    '{emoji}': ['🤷', '👍', '👌', '😶', '😑', '🤔']
}
neutral_base = [
    "This is fine ig 🤷", "Ur post seems okay tbh 👍", "Thanks for sharing this 👌",
    "Looks chill to me 😶", "I get ur point fr", "This idea is meh ngl 🤷",
    "Seems decent, no shade", "Just a cool vibe ig", "I see what u mean tbh",
    "This is alright, no cap 👍", "behave in right way 😶", "just saying, seems fine 👌",
    "no big deal fr 🤷", "this is chill tbh 😶", "all good ig 😑", "no comment fr 🤔"
]

# Invalid examples
invalid_templates = [
    "{random_text}", "{punctuation}", "{short_text}"
]
invalid_replacements = {
    '{random_text}': KEYWORD_RULES['Invalid'][:10],
    '{punctuation}': ['!!!', '@#$%^', '***', '???', '!!!???', '!!!!', '**##@@', '+++='],
    '{short_text}': ['', 'a', '1', '@', '##', '#']
}
invalid_base = KEYWORD_RULES['Invalid']

# Generate full datasets
positive_comments = generate_examples(positive_base, positive_templates, positive_replacements)
negative_comments = generate_examples(negative_base, negative_templates, negative_replacements)
sarcastic_comments = generate_examples(sarcastic_base, sarcastic_templates, sarcastic_replacements)
risky_comments = generate_examples(risky_base, risky_templates, risky_replacements)
neutral_comments = generate_examples(neutral_base, neutral_templates, neutral_replacements)
invalid_comments = generate_examples(invalid_base, invalid_templates, invalid_replacements, target_count=3000)

# Create DataFrame
data = {
    'text': (positive_comments + negative_comments + sarcastic_comments +
             risky_comments + neutral_comments + invalid_comments),
    'label': (['Positive']*5000 + ['Negative']*5000 + ['Sarcastic']*5000 +
              ['Risky']*5000 + ['Neutral']*5000 + ['Invalid']*3000)
}
df = pd.DataFrame(data)

# Preprocessing
def clean_text(text):
    text = str(text).lower().strip()
    text = emoji.demojize(text)
    text = re.sub(r'[^\w\s:*…😀-🙏]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)
df['vader_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(emoji.emojize(x))['compound'])

# Feature extraction
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words='english', min_df=5)
X_tfidf = vectorizer.fit_transform(df['clean_text'])
X = np.hstack((X_tfidf.toarray(), df[['vader_score']].values))

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=200, max_depth=30, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
logger.info("\nClassification Report:\n")
logger.info(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save artifacts
joblib.dump(model, 'models/sentiment_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

# Test with real-world examples
test_comments = [
    "fuck you bab",
    "you are a lier",
    "i not accept what are you saying",
    "behave in right way",
    "you are good boy",
    "you are waste of my time",
    "You clearly don’t know what you're doing 🤦",
    "I will kill you 😡",
    "why are you so illiterate",
    "how are you pokie",
    "asdjkl",
    "!!!",
    "congraulations bro! 🎉",
    "why u so dumb fr? 😒",
    "this is so lit 🔥",
    "sure, great idea… 🙄",
    "watch ur back 😈",
    "meh, its whatever 🤷",
    "ur post is garbage 😩",
    "youre a legend! 💯"
]
logger.info("\n Testing Real-World Social Media Comments:\n")
for comment in test_comments:
    cleaned_text = clean_text(comment)
    vader_score = analyzer.polarity_scores(emoji.emojize(cleaned_text))['compound']
    X_new = vectorizer.transform([cleaned_text]).toarray()
    X_new = np.hstack((X_new, [[vader_score]]))
    probs = model.predict_proba(X_new)[0]
    pred = model.predict(X_new)
    predicted_label = label_encoder.inverse_transform(pred)[0]
    logger.info(f"📝 Comment: {emoji.emojize(comment)}\n🔮 Predicted Sentiment: {predicted_label}\n{'-'*60}")
