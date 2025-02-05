import streamlit as st
import pickle
import string
import nltk
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Set NLTK data directory
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):  # Create directory if it doesn't exist
    os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)  # Ensure Streamlit finds nltk data

# Download necessary NLTK resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# Ensure correct tokenizer is found
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


vectorizer_path = os.path.join(os.getcwd(), 'vectorizer.pkl')
model_path = os.path.join(os.getcwd(), 'model.pkl')

try:
    tfidf = pickle.load(open(vectorizer_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()
    
st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("🚨 Spam 🚨")
    else:
        st.header("✅ Not Spam ✅")
