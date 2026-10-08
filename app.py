import streamlit as st
import pickle
import re

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Spam SMS Classifier",
    page_icon="📩",
    layout="centered"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    /* Hide Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main background - gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        max-width: 850px;
    }
    
    /* Title - WHITE for gradient bg */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-weight: 900 !important;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 2px 15px rgba(0, 0, 0, 0.4);
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #f0f0f0 !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Subheaders - BLACK */
    h3 {
        color: #333333 !important;
        font-weight: 700 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    [data-testid="stMetricValue"] {
        color: #667eea !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666 !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        color: #000000 !important;
        background: #ffffff !important;
        border-radius: 15px !important;
        border: 2px solid #e0e0e0 !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #999999 !important;
        opacity: 1 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* BUTTON - PURPLE */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.9rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        color: #ffffff !important;
    }
    
    .stButton > button p {
        color: #ffffff !important;
    }
    
    /* Warning box - YELLOW */
    div[data-testid="stAlert"] {
        background-color: #fff3cd !important;
        border-radius: 15px !important;
        border-left: 5px solid #ffc107 !important;
        padding: 1.2rem !important;
    }
    
    div[data-testid="stAlert"] p {
        color: #856404 !important;
        font-weight: 600 !important;
    }
    
    /* Success box - GREEN */
    div[data-baseweb="notification"]:has(div[data-testid="stMarkdownContainer"]) {
        border-radius: 15px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #333333 !important;
        font-weight: 600 !important;
    }
    
    /* Paragraphs - dark */
    p {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============ LOAD MODEL ============
@st.cache_resource
def load_model():
    model = pickle.load(open('spam_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_model()

# ============ STOPWORDS ============
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
    'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn',
    'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
    'weren', 'won', 'wouldn'
])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# ============ HEADER ============
st.markdown("# 📩 Spam SMS Classifier")
st.markdown('<p class="subtitle">AI-powered spam detection in real-time</p>', unsafe_allow_html=True)

st.markdown("---")

# ============ METRICS ============
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="🎯 Accuracy", value="97.22%")
with col2:
    st.metric(label="📊 Trained On", value="5572 msgs")
with col3:
    st.metric(label="🚀 Precision", value="0.99")

st.markdown("---")

# ============ INPUT ============
st.markdown("### ✍️ Enter Your Message")

user_input = st.text_area(
    "Message:",
    height=150,
    placeholder="Type or paste your SMS message here...\n\nExample: WINNER! You have won £1000 cash! Call now!",
    label_visibility="collapsed"
)

if st.button("🔍  Analyze Message"):
    if user_input.strip() == "":
        st.warning("⚠️  Please enter a message first!")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]
        
        st.markdown("---")
        st.markdown("### 📊 Result")
        
        if prediction == 1:
            conf = probability[1] * 100
            st.error(f"🚨 **SPAM DETECTED!**\n\nThis message is spam.\n\n**Confidence:** {conf:.2f}%")
        else:
            conf = probability[0] * 100
            st.success(f"✅ **NOT SPAM (HAM)**\n\nThis message is safe.\n\n**Confidence:** {conf:.2f}%")
        
        with st.expander("🔍 View Details"):
            st.write(f"**📝 Original:** {user_input}")
            st.write(f"**🧹 Cleaned:** {cleaned if cleaned else '(empty)'}")

# ============ FOOTER ============
st.markdown("---")
st.markdown("<p style='text-align: center; color: #999; font-size: 0.9rem;'>Built with ❤️ using Python + Scikit-learn + Streamlit</p>", unsafe_allow_html=True)