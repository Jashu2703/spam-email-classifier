import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords')

def preprocess(text):
        text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

#directly link an dataset or link an downloaded dataset spam.csv
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', header=None)
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df['message'] = df['message'].apply(preprocess)

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

def predict_email(text):
    processed = preprocess(text)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)
    return "Spam" if prediction[0] == 1 else "Ham"

custom_email = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."
print("Prediction:", predict_email(custom_email))
