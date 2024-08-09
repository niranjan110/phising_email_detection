import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Load the dataset
df = pd.read_csv('Phishing_Email.csv')

# Handle NaN values and preprocess the data
df = df.dropna(subset=['Email Text'])
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(df['Email Text'])
y = df['Email Type']

# Train the model
model = MultinomialNB()
model.fit(X, y)

# Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')

print("Model and vectorizer saved successfully.")
