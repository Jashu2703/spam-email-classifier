import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
import string
from nltk.corpus import stopwords

# Download stopwords if not already
nltk.download('stopwords')

# Preprocessing function
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', header=None)
df.columns = ['label', 'message']

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Preprocess messages
df['message'] = df['message'].apply(preprocess)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Try with custom email
def predict_email(text):
    processed = preprocess(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example
custom_email = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."
print("Prediction:", predict_email(custom_email))
