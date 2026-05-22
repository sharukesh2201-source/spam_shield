from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

texts = [
    "Win free money",
    "Claim free iPhone",
    "Meeting tomorrow",
    "Call me later",
    "Lottery winner",
    "How are you"
]

labels = [
    "spam",
    "spam",
    "safe",
    "safe",
    "spam",
    "safe"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Trained")