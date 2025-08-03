import streamlit as st
import re
import string
import pickle
import nltk
import numpy as np
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.stem import SnowballStemmer

# Download NLTK data
nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Load model and tokenizer
model = load_model("model.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split() if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

# Streamlit UI
st.title("🛡️ Hate Speech Detection")
st.markdown("Enter any sentence below to check if it contains hate/abusive language.")

user_input = st.text_area("🔍 Enter your text here:", height=150)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=300)
        pred = model.predict(padded)[0][0]

        st.subheader("🔎 Result:")
        if pred < 0.5:
            st.success("✅ No hate or abusive language detected.")
        else:
            st.error("⚠️ Hate or abusive content detected.")
        st.write(f"**Prediction Score:** `{pred:.4f}`")
