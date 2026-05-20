import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import os
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import tensorflow as tf
from gensim.models import Word2Vec

# --- 1. SETUP PAGE & CACHING ---
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="🐦", layout="wide")

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')

@st.cache_resource
def init_nlp_tools():
    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    
    factory_stop = StopWordRemoverFactory()
    stopwords_id = factory_stop.get_stop_words()
    
    return stemmer, stopwords_id

download_nltk_data()
stemmer, stopwords_id = init_nlp_tools()

# Kamus Slang sesuai notebook
slang_dict = {
    "gk": "tidak", "gak": "tidak", "ga": "tidak", "yg": "yang",
    "kalo": "kalau", "klo": "kalau", "dgn": "dengan", "bgt": "banget",
    "gpp": "tidak apa apa", "udh": "sudah", "udah": "sudah",
    "aja": "saja", "tp": "tapi", "jd": "jadi", "drpd": "daripada",
    "kek": "seperti", "nder": "sender", "wkwk": "", "wkwkwk": ""
}

# --- 2. PREPROCESSING FUNCTION ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[USERNAME\]|\[URL\]|\[HASHTAG\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    tokens = word_tokenize(text)
    tokens = [slang_dict.get(word, word) for word in tokens]
    tokens = [word for word in tokens if word.strip() != '']
    tokens = [word for word in tokens if word not in stopwords_id]
    
    stemmed_words = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_words)

# --- 3. WORD2VEC HELPER FUNCTIONS ---
def get_w2v_average(tokens, model, vector_size=100):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def get_w2v_sequence(tokens, model, vector_size=100, max_len=50):
    vec_seq = [model.wv[word] if word in model.wv else np.zeros(vector_size) for word in tokens]
    if len(vec_seq) < max_len:
        padding = [np.zeros(vector_size)] * (max_len - len(vec_seq))
        vec_seq = padding + vec_seq
    else:
        vec_seq = vec_seq[:max_len]
    return np.array(vec_seq)

# --- 4. LOAD MODELS ---
@st.cache_resource
def load_models():
    models = {}
    base_path = 'saved_models'
    
    # Load Label Encoder
    try:
        with open(f'{base_path}/label_encoder.pkl', 'rb') as f:
            models['le'] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading Label Encoder: {e}")
        
    # Load RF Pipelines
    rf_files = {'TF-IDF': 'Pipeline_RF_TF-IDF.pkl', 'N-Gram': 'Pipeline_RF_N-Gram.pkl'}
    for feature, file in rf_files.items():
        try:
            with open(f'{base_path}/{file}', 'rb') as f:
                models[f'RF_{feature}'] = pickle.load(f)
        except FileNotFoundError:
            pass # Handle gracefully later
            
    # Load Word2Vec & DL Vectorizers
    try:
        models['w2v_model'] = Word2Vec.load(f'{base_path}/word2vec_model.bin')
        with open(f'{base_path}/tfidf_vectorizer_dl.pkl', 'rb') as f:
            models['tfidf_dl'] = pickle.load(f)
        with open(f'{base_path}/ngram_vectorizer_dl.pkl', 'rb') as f:
            models['ngram_dl'] = pickle.load(f)
    except Exception:
        pass

    # Load BiLSTM Models
    dl_features = ['TF-IDF', 'N-Gram', 'Word2Vec']
    for feature in dl_features:
        try:
            models[f'BiLSTM_{feature}'] = tf.keras.models.load_model(f'{base_path}/BiLSTM_{feature}.keras')
        except Exception:
            pass
            
    return models

models = load_models()

# --- 5. UI LAYOUT & LOGIC ---
st.title("🐦 Analisis Sentimen Twitter")
st.markdown("Aplikasi pintar untuk mengklasifikasikan sentimen tweet berbahasa Indonesia menjadi **Positif**, **Netral**, atau **Negatif** menggunakan Machine Learning dan Deep Learning.")

st.sidebar.header("⚙️ Konfigurasi Model")
selected_model = st.sidebar.selectbox("Pilih Arsitektur Model:", ["Random Forest", "BiLSTM"])
selected_feature = st.sidebar.selectbox("Pilih Ekstraksi Fitur:", ["TF-IDF", "N-Gram", "Word2Vec"])

st.sidebar.divider()
st.sidebar.info("Pastikan folder `saved_models` berisi file model (.pkl, .bin, .keras) hasil training Anda diletakkan satu folder dengan aplikasi ini.")

# Input text area
tweet_input = st.text_area("Masukkan teks tweet di sini:", placeholder="Contoh: Aplikasi ini sangat membantu dan UI-nya keren banget! gk nyesel pake ini.")

if st.button("Analisis Sentimen", type="primary"):
    if not tweet_input.strip():
        st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        with st.spinner("Memproses teks dan melakukan prediksi..."):
            # 1. Preprocessing
            clean_text = preprocess_text(tweet_input)
            
            prediction = None
            model_key = f"{selected_model.replace(' ', '')}_{selected_feature}"
            
            try:
                # 2. Eksekusi Prediksi Berdasarkan Pilihan
                if selected_model == "Random Forest":
                    if selected_feature in ["TF-IDF", "N-Gram"]:
                        rf_pipeline = models.get(f'RF_{selected_feature}')
                        if rf_pipeline:
                            pred_idx = rf_pipeline.predict([clean_text])[0]
                            prediction = models['le'].inverse_transform([pred_idx])[0]
                        else:
                            st.error(f"Model RF dengan {selected_feature} tidak ditemukan di saved_models.")
                    
                    elif selected_feature == "Word2Vec":
                        # Logika manual jika RF Word2Vec disimpan terpisah
                        st.warning("Pipeline Random Forest + Word2Vec tidak di-export di notebook awal. Pastikan Anda telah melatih dan menyimpannya.")
                
                elif selected_model == "BiLSTM":
                    bilstm_model = models.get(f'BiLSTM_{selected_feature}')
                    if not bilstm_model:
                        st.error(f"Model BiLSTM {selected_feature} tidak ditemukan.")
                    else:
                        if selected_feature == "TF-IDF":
                            vec = models['tfidf_dl'].transform([clean_text]).toarray()
                            vec = np.expand_dims(vec, axis=1) # (1, 1, Features)
                        elif selected_feature == "N-Gram":
                            vec = models['ngram_dl'].transform([clean_text]).toarray()
                            vec = np.expand_dims(vec, axis=1)
                        elif selected_feature == "Word2Vec":
                            tokens = clean_text.split()
                            vec = get_w2v_sequence(tokens, models['w2v_model'], 100, 50)
                            vec = np.expand_dims(vec, axis=0) # (1, 50, 100)
                            
                        pred_prob = bilstm_model.predict(vec)
                        pred_idx = np.argmax(pred_prob, axis=1)[0]
                        prediction = models['le'].inverse_transform([pred_idx])[0]

                # 3. Menampilkan Hasil
                if prediction:
                    st.divider()
                    st.subheader("Hasil Analisis")
                    st.write(f"**Teks Asli:** {tweet_input}")
                    st.write(f"**Teks Bersih (Preprocessed):** `{clean_text}`")
                    
                    # Logika warna berdasarkan label
                    if str(prediction).lower() == 'positive':
                        st.success(f"🌟 Sentimen: **{prediction.upper()}**")
                    elif str(prediction).lower() == 'negative':
                        st.error(f"🚨 Sentimen: **{prediction.upper()}**")
                    else:
                        st.info(f"⚖️ Sentimen: **{prediction.upper()}**")
                        
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses: {e}")