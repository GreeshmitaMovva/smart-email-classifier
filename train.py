"""Train a small demonstration email text classifier."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "email_model.joblib"

# Small built-in sample set keeps the demo self-contained.
# Replace this with a larger, properly labelled dataset for meaningful results.
EMAILS = [
    "Meeting scheduled for tomorrow at ten",
    "Please find the project report attached",
    "Your assignment feedback is available",
    "The team meeting has been moved to Monday",
    "Thank you for your application",
    "Your order confirmation is ready",
    "Here are the notes from our lecture",
    "Please review the document before Friday",
    "The exam timetable has been published",
    "Your appointment is confirmed",
    "I have shared the presentation with the team",
    "The class starts at nine in the morning",
    "Your payment receipt is attached",
    "Can we reschedule our meeting for Tuesday",
    "The requested document is ready for review",
    "Win a free prize claim your reward now",
    "Congratulations you have won free cash",
    "Urgent claim your lottery winnings today",
    "Click here to receive your free gift",
    "You won a gift card claim immediately",
    "Get rich quickly with this secret offer",
    "Act now to claim your cash reward",
    "Your account won a free bonus prize",
    "Limited offer claim your reward today",
    "You are selected to receive free money",
    "Claim your exclusive prize before it expires",
    "Free vacation waiting click to claim now",
    "You have won a cash reward act today",
    "Congratulations claim your lottery prize",
    "Earn money fast with this special offer",
    "Your free reward is waiting for you",
    "Claim bonus cash immediately using this link",
]

LABELS = [
    "legitimate", "legitimate", "legitimate", "legitimate",
    "legitimate", "legitimate", "legitimate", "legitimate",
    "legitimate", "legitimate", "legitimate", "legitimate",
    "legitimate", "legitimate", "legitimate",
    "spam_phishing", "spam_phishing", "spam_phishing",
    "spam_phishing", "spam_phishing", "spam_phishing",
    "spam_phishing", "spam_phishing", "spam_phishing",
    "spam_phishing", "spam_phishing", "spam_phishing",
    "spam_phishing", "spam_phishing", "spam_phishing",
    "spam_phishing", "spam_phishing",
]


def train_model():
    """Fit and save the demo model; return the fitted pipeline."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(EMAILS, LABELS)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_model()
