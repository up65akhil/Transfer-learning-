import streamlit as st
from transformers import pipeline
import time

st.set_page_config(
    page_title="Sentiment Analysis", 
    page_icon="🔮", 
    layout="centered",
    initial_sidebar_state="expanded"
) 
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103140.png", width=80)
    st.markdown("## System Status")
    st.success("🟢 Neural Network Online")
    st.markdown("---")
    st.markdown("### 🧠 How it Works")
    st.markdown("This AI uses a **DistilBERT Deep Learning Transformer**. It reads your text forward and backward simultaneously to understand context, sarcasm, and slang.")
    st.markdown("*Fine-tuned on 10,000+ real-world interactions.*")

@st.cache_resource
def load_model():
    model_folder = "up65akhil/up65akhil-sentiment-model"
    return pipeline("text-classification", model=model_folder, tokenizer=model_path)

try:
    sentiment_pipeline = load_model()
except Exception as e:
    st.error(f"System Offline: Unable to load model weights. Please check directory structure.")
    st.stop()

st.markdown('<div class="neon-text">Sentiment AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Advanced Contextual Sentiment Analysis</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="Talk to the AI:", 
    label_visibility="collapsed",
    height=120, 
    placeholder="Type something here, and the AI will tell you how it feels... (e.g., The customer service was amazing!)"
)

colA, colB = st.columns(2)
if colA.button("📝 Try Sarcasm Example"):
    user_input = "I am absolutely thrilled that my flight was delayed for 6 hours."
    st.rerun()
if colB.button("🗑️ Clear Text"):
    user_input = ""
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    
    if user_input.strip() == "":
        st.toast('Please enter some text first!', icon='⚠️')
    else:
        with st.status("Initializing Neural Pathways...", expanded=True) as status:
            st.write("Tokenizing input text...")
            time.sleep(0.5) 
            st.write("Running through 6 transformer layers...")
            time.sleep(0.5)
            
            result = sentiment_pipeline(user_input)[0]
            raw_label = result['label']
            confidence = result['score']
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
        st.markdown("### 🤖 AI Response:")
        
        if raw_label == "LABEL_0":
            st.markdown('<div class="result-card neg-card">📉 Negative Sentiment Detected</div>', unsafe_allow_html=True)
            st.write(f"**The AI says:** *\"Ouch. This sounds pretty frustrated or upset. I am picking up a very negative tone here.\"*")
            st.progress(confidence, text=f"Confidence Level: {confidence:.1%}")
            
        elif raw_label == "LABEL_1":
            st.markdown('<div class="result-card neu-card">➖ Neutral Sentiment Detected</div>', unsafe_allow_html=True)
            st.write(f"**The AI says:** *\"Very factual and straight to the point. This reads as completely neutral to me.\"*")
            st.progress(confidence, text=f"Confidence Level: {confidence:.1%}")
            
        elif raw_label == "LABEL_2":
            st.markdown('<div class="result-card pos-card">📈 Positive Sentiment Detected</div>', unsafe_allow_html=True)
            st.write(f"**The AI says:** *\"Wow! That is great energy. I am picking up a very positive, happy vibe from this text!\"*")
            st.progress(confidence, text=f"Confidence Level: {confidence:.1%}")
            
            if confidence > 0.90:
                st.balloons()
