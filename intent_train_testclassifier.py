from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import streamlit as st

# Load preprocessed data
data = pd.read_csv('preprocessed_intent_data.csv')

# Verify unique intent labels
unique_intents = data['intent'].unique()

# Ensure you have multiple unique intent labels
if len(unique_intents) <= 1:
    raise ValueError("You need at least two unique intent labels for classification.")

# Initialize TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the preprocessed sentences
tfidf_matrix = tfidf_vectorizer.fit_transform(data['preprocessed_sentence'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, data['intent'], test_size=0.3, random_state=42)

# Initialize SVM classifier
clf = SVC()

# Train the model
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'intent_classifier_model.pkl')

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Streamlit app
st.title("Intent Classification App")

intent_scraptWord = ""

# Text input for user to enter a sentence
user_age = st.text_input("Enter your age:", "")

if user_age < 7 :
    intent_scraptWord += "Baby "
elif user_age < 18:
    intent_scraptWord += "Teen "
elif user_age < 55:
    intent_scraptWord += "Young "
else:
    intent_scraptWord += "Old "
    
# Text input for user to enter a sentence
user_gender = st.text_input("Enter your gender:", "")
intent_scraptWord += user_gender

# Text input for user to enter a sentence
user_input = st.text_input("Enter occassion:", "")

user_occassion = ""

# Predict intent on user input
if user_input:
    # Transform user input using the same TF-IDF vectorizer
    user_input_vector = tfidf_vectorizer.transform([user_input])

    # Make prediction
    user_occassion = clf.predict(user_input_vector)[0]
    
else:
    st.write("Please enter a sentence to predict its intent.")
    
intent_scraptWord += user_occassion
intent_scraptWord += " cloth"