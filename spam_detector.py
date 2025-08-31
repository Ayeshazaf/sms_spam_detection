import re  
import joblib
import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import streamlit as st
import warnings 
warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

# Download latest version

def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)             # remove @mentions
    text = re.sub(r'[^a-z\s]', ' ', text)        # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text
@st.cache_data
def load_data(path="spamdata/spam_sms.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"v1": "kind", "v2": "text"})
    df["CleanMessage"] = df["text"].apply(cleaning_data) 
    return df

@st.cache_resource
def feature_extraction(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # no need stop_words, we already cleaned
    X = vectorizer.fit_transform(df["CleanMessage"])
    encoder = LabelEncoder()
    Y = encoder.fit_transform(df["kind"])  # 0 = ham, 1 = spam
    joblib.dump(vectorizer, "vectorizer.joblib")
    joblib.dump(encoder, "encoder.joblib")
    return X, Y, vectorizer, encoder


#Spliting data into train/test sets.
def split_data(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

#Train a classifier Naive Bayes, logistic regression

def train_classifier(x_train, y_train):
    model_naive = MultinomialNB().fit(x_train, y_train)
    model_reg = LogisticRegression(max_iter=200,class_weight="balanced").fit(x_train, y_train)
    joblib.dump(model_naive, "naive_model.joblib")
    joblib.dump(model_reg, "logreg_model.joblib")
    return model_naive , model_reg

#Evaluate accuracy, precision, recall, F1-score on test set.

def evaluate_model(model, x_test, y_test): 
    y_pred = model.predict(x_test) 
    accuracy = accuracy_score(y_test, y_pred) 
    precision = precision_score(y_test, y_pred, average='weighted') 
    recall = recall_score(y_test, y_pred, average='weighted') 
    f1 = f1_score(y_test, y_pred, average='weighted') 
    return accuracy, precision, recall, f1

# Write a function to predict if a new message is spam or not.

def predict_spam(model, vectorizer, message):
    cleaned_message = cleaning_data(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction = model.predict(vectorized_message)
    return prediction

# Build Streamlit UI Input box for user to enter a message. Display prediction (spam/ham).
def main():
    st.title("Spam Detector") 
    df = load_data()  
    # Prepare models only once
    X, Y, vectorizer, encoder = feature_extraction(df)
    x_train, x_test, y_train, y_test = split_data(X, Y)
    if os.path.exists("naive_model.joblib") and os.path.exists("logreg_model.joblib") and os.path.exists("vectorizer.joblib"):
        model_naive = joblib.load("naive_model.joblib")
        model_reg = joblib.load("logreg_model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
    else:
    # Train and save as above
        model_naive, model_reg = train_classifier(x_train, y_train)

    # Evaluate both models
    acc_naive, prec_naive, rec_naive, f1_naive = evaluate_model(model_naive, x_test, y_test)
    acc_reg, prec_reg, rec_reg, f1_reg = evaluate_model(model_reg, x_test, y_test)

    left, right = st.columns([3,1])
    with right:
        st.subheader("📊 Model Performance")
        st.write("Naive Bayes → Accuracy:", round(acc_naive, 3), "F1:", round(f1_naive, 3))
        st.write("Logistic Regression → Accuracy:", round(acc_reg, 3), "F1:", round(f1_reg, 3))

        # Pick best model automatically
        model = model_naive if acc_naive > acc_reg else model_reg
        best_model = "Naive Bayes" if acc_naive > acc_reg else "Logistic Regression"
        st.success(f"Best model selected: **{best_model}**")
    with left:
    # User input
        user_input = st.text_area("Enter your message:")
        if st.button("Predict"):
            prediction = predict_spam(model, vectorizer, user_input)
            st.write("Prediction:", "Spam" if prediction[0] else "Ham")
main()