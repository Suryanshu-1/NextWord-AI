import os
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_model (1).h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LEN_PATH = os.path.join(BASE_DIR, "max_len.pkl")


@st.cache_resource
def load_next_word_assets():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")
    if not os.path.exists(MAX_LEN_PATH):
        raise FileNotFoundError(f"Max length file not found: {MAX_LEN_PATH}")

    with open(TOKENIZER_PATH, "rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    with open(MAX_LEN_PATH, "rb") as max_len_file:
        max_len = int(pickle.load(max_len_file))

    model = load_model(MODEL_PATH)
    return model, tokenizer, max_len


def get_top_predictions(text, top_n=1):
    model, tokenizer, max_len = load_next_word_assets()

    cleaned_text = " ".join(str(text).strip().split())
    if not cleaned_text:
        return []

    token_list = tokenizer.texts_to_sequences([cleaned_text])[0]
    if not token_list:
        return []

    last_seq = token_list[-max_len:]
    padded_seq = pad_sequences([last_seq], maxlen=max_len, padding="pre")

    prediction = model.predict(padded_seq, verbose=0)[0]
    top_indices = np.argsort(prediction)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        word = tokenizer.index_word.get(int(idx), "")
        if word:
            results.append(word)

    return results[:top_n]


st.set_page_config(
    page_title="🔮 NextWord AI",
    page_icon="✨",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #111827 40%, #1e293b 100%);
            color: #e2e8f0;
        }
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #111827 40%, #1e293b 100%);
        }
        .title-box {
            padding: 1.25rem 1.5rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.2);
        }
        .prediction-card {
            padding: 1rem 1.25rem;
            border-radius: 16px;
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.25);
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .block-spacer {
            height: 1rem;
        }
        .big-label {
            font-size: 2.15rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .small-note {
            color: #a5b4fc;
            font-size: 0.95rem;
        }
        div[data-testid="stTextArea"] textarea {
            font-size: 1.05rem;
            color: #e2e8f0;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 14px;
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(96, 165, 250, 0.35);
            background: linear-gradient(135deg, #2563eb 0%, #4338ca 100%);
            color: white;
            font-weight: 600;
            padding: 0.65rem 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-box">
        <div class="big-label">🔮 NextWord AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="block-spacer"></div>', unsafe_allow_html=True)

if "typed_text" not in st.session_state:
    st.session_state.typed_text = "The future of AI is"

if "predictions" not in st.session_state:
    st.session_state.predictions = []


def append_word_to_sentence(word):
    current_text = st.session_state.get("typed_text", "")
    current_text = current_text.strip()
    if not current_text:
        st.session_state.typed_text = word.strip()
    else:
        st.session_state.typed_text = f"{current_text} {word.strip()}".strip()
    st.session_state.predictions = []


with st.form("sentence_form"):
    text_input = st.text_area(
        "Sentence",
        value=st.session_state.typed_text,
        key="typed_text",
        height=220,
        help="Keep typing a phrase to get smart next-word suggestions from the trained model.",
    )

    predict_clicked = st.form_submit_button("Predict next word", use_container_width=True)

    if predict_clicked:
        try:
            st.session_state.predictions = get_top_predictions(text_input, top_n=1)
        except Exception as exc:
            st.session_state.predictions = []
            st.error(f"Prediction failed: {exc}")

st.markdown('<div class="block-spacer"></div>', unsafe_allow_html=True)

if st.session_state.predictions:
    word = st.session_state.predictions[0]
    st.markdown(
        f"""
        <div class="prediction-card" style="background: rgba(22, 163, 74, 0.18); border: 1px solid rgba(74, 222, 128, 0.5); color: #dcfce7; font-size: 1.5rem; font-weight: 600; text-align: center;">
            {word}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="prediction-card">
            Start typing a sentence and press <b>Predict next word</b> to see the predicted word.
        </div>
        """,
        unsafe_allow_html=True,
    )
