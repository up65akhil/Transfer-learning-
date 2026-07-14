import streamlit as st
from transformers import pipeline
import time
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Sentiment Analysis AI", 
    page_icon="🔮", 
    layout="centered",
    initial_sidebar_state="expanded"
) 

# 2. Safely load custom CSS without crashing if the file is missing
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
else:
    st.warning("⚠️ `style.css` not found. Running with default Streamlit styling.")

# 3. Initialize Session State for interactive buttons
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# 4. Sidebar UI
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103140.png", width=80)
    st.markdown("## System Status")
    st.success("🟢 Neural Network Online")
    st.markdown("---")
    st.markdown("### 🧠 How it Works")
    st.markdown("This AI uses a **DistilBERT Deep Learning Transformer** to analyze contextual sentiment, sarcasm, and tone.")

# 5. CRITICAL FIX: @st.cache_resource prevents memory crashes and infinite reload loops
@st.cache_resource(show_spinner="Downloading & initializing AI weights into system memory...")
def load_model():
    # Using a lightweight, pre-trained sentiment model optimized for 1GB cloud RAM limits
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    return pipeline("text-classification", model=model_name, tokenizer=model_name)

# Safely initialize the pipeline with UI error handling
try:
    sentiment_pipeline = load_model()
except Exception as e:
    st.error(f"⚠️ System Offline: Unable to initialize neural network weights. Error details: {str(e)}")
    st.stop()

# 6. Header UI
st.markdown('<div class="neon-text">Sentiment AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Advanced Contextual Sentiment Analysis</div>', unsafe_allow_html=True)

# Helper callbacks for interactive example buttons
def set_sarcasm_example():
    st.session_state.text_input = "I am absolutely thrilled that my flight was delayed for 6 hours."

def clear_text():
    st.session_state.text_input = ""

# Action Buttons
col1, col2 = st.columns(2)
col1.button("📝 Try Sarcasm Example", on_click=set_sarcasm_example, use_container_width=True)
col2.button("🗑️ Clear Text", on_click=clear_text, use_container_width=True)

# Text Input Area
user_input = st.text_area(
    label="Talk to the AI:", 
    label_visibility="collapsed",
    height=130, 
    key="text_input",
    placeholder="Type a sentence or review here... (e.g., The customer service was amazing!)"
)

st.markdown("<br>", unsafe_allow_html=True)

# 7. Prediction & Inference Logic
if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not user_input.strip():
        st.toast("Please enter some text first!", icon="⚠️")
    else:
        with st.status("Analyzing neural pathways...", expanded=True) as status:
            st.write("Tokenizing input text...")
            time.sleep(0.2) 
            st.write("Processing through transformer layers...")
            time.sleep(0.2)
            
            # Run inference
            result = sentiment_pipeline(user_input)[0]
            raw_label = str(result['label']).upper()
            confidence = float(result['score'])
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
        st.markdown("### 🤖 AI Response:")
        safe_confidence = min(max(confidence, 0.0), 1.0)
        
        # Flexible label matching to handle different Hugging Face output formats
        if any(neg_tag in raw_label for neg_tag in ["LABEL_0", "NEG", "0"]):
            st.markdown('<div class="result-card neg-card">📉 Negative Sentiment Detected</div>', unsafe_allow_html=True)
            st.write('**The AI says:** *"This sounds frustrated or upset. I am picking up a negative tone here."*')
            st.progress(safe_confidence, text=f"Confidence Level: {safe_confidence:.1%}")
            
        elif any(pos_tag in raw_label for pos_tag in ["LABEL_1", "POS", "1", "LABEL_2"]):
            st.markdown('<div class="result-card pos-card">📈 Positive Sentiment Detected</div>', unsafe_allow_html=True)
            st.write('**The AI says:** *"Great energy! I am picking up a very happy, positive vibe from this text!"*')
            st.progress(safe_confidence, text=f"Confidence Level: {safe_confidence:.1%}")
            if safe_confidence > 0.88:
                st.balloons()
        else:
            st.markdown('<div class="result-card neu-card">➖ Neutral Sentiment Detected</div>', unsafe_allow_html=True)
            st.write('**The AI says:** *"Very factual and straightforward. This reads as completely neutral."*')
            st.progress(safe_confidence, text=f"Confidence Level: {safe_confidence:.1%}")
