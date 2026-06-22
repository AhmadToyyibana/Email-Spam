import streamlit as st
import joblib

# Load model dan tfidf
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("📧 Email Spam Classifier")
st.write("Implementasi Multinomial Naive Bayes dengan TF-IDF")

email = st.text_area("Masukkan isi email")

if st.button("Prediksi"):
    vector = tfidf.transform([email])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.error("🚨 SPAM")
    else:
        st.success("✅ HAM (Bukan Spam)")
