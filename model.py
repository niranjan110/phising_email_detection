import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load and preprocess the dataset
df = pd.read_csv('Phishing_Email.csv')

# Fill any missing values in the text column with an empty string
df['text'] = df['text'].fillna('')

# Separate features and labels
X = df['text']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data, and transform the testing data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]  # Probability of being phishing
print(y_pred)
print(y_pred_proba)  # Print the probabilities for each email

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and the TF-IDF vectorizer to files
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')

print("Model and vectorizer have been saved as 'model.pkl' and 'vectorizer.pkl'.")
