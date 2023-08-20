

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from categoriesMapping import myDictionary
from trends_features import *
from personalised_results import *



# Loading preprocessed data, verifying labels, ensuring multiple unique label, fitting and transforming
# model intrial svm classifier and training model, saving model, evaluating model, 
data = pd.read_csv('preprocessed_intent_data.csv')

unique_intents = data['intent'].unique()

if len(unique_intents) <= 1:
    raise ValueError("You need at least two unique intent labels for classification.")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['preprocessed_sentence'])

X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, data['intent'], test_size=0.3, random_state=42)
clf = SVC()

clf.fit(X_train, y_train)
joblib.dump(clf, 'intent_classifier_model.pkl')
y_pred = clf.predict(X_test)


def intentOutput(user_input):
        # Predict intent on user input
    if user_input:
        user_input_vector = tfidf_vectorizer.transform([user_input])

        # Make prediction from intent classification
        return clf.predict(user_input_vector)[0]
        
    else:
        print("Please enter a sentence to predict its intent.")
