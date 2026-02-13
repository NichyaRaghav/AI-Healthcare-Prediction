import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="MedAI Pro", layout="wide")

# ---------------- BACKGROUND & STYLE ----------------
st.markdown("""
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1588776814546-1ffcf47267a5");
    background-size: cover;
}
.card {
    background-color: rgba(255,255,255,0.92);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    margin-bottom: 15px;
}
.title {
    text-align:center;
    color:#c62828;
    font-size:42px;
    font-weight:800;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>💓 MedAI Pro – AI Health Assistant</div>", unsafe_allow_html=True)

# ---------------- DISEASE DATA ----------------
DISEASES = [
    {"name": "Diabetes", "symptoms": "frequent urination thirst fatigue", "doctor": "Endocrinologist", "precautions": "Avoid sugar, regular exercise"},
    {"name": "Heart Attack", "symptoms": "chest pain sweating nausea", "doctor": "Cardiologist", "precautions": "🚨 Call 108 immediately"},
    {"name": "Flu", "symptoms": "fever cough sore throat", "doctor": "General Physician", "precautions": "Rest, fluids, paracetamol"},
    {"name": "Asthma", "symptoms": "wheezing breathlessness chest tightness", "doctor": "Pulmonologist", "precautions": "Avoid triggers, use inhaler"},
    {"name": "Hypertension", "symptoms": "headache dizziness blurred vision", "doctor": "Cardiologist", "precautions": "Low salt diet, BP check"},
    {"name": "Pneumonia", "symptoms": "fever cough chest pain breathlessness", "doctor": "Pulmonologist", "precautions": "Hospital consultation"}
]

df = pd.DataFrame(DISEASES)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(df["symptoms"])

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- SIDEBAR LOGIN ----------------
with st.sidebar:
    st.header("🔐 Login")
    if not st.session_state.logged_in:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u and p:
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
    else:
        st.success("Welcome " + st.session_state.user)
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

# ---------------- MAIN APP ----------------
if st.session_state.logged_in:

    st.error("🚨 Chest pain or breathing difficulty? Call 108 immediately!")

    tab1, tab2 = st.tabs(["🔍 Symptom Analyzer", "📊 Quick Scan"])

    # ---------- ANALYZER ----------
    with tab1:
        st.subheader("📝 Enter Your Symptoms")
        user_symptoms = st.text_area("Example: fever, cough, chest pain")

        if st.button("Analyze Symptoms ❤️"):
            if user_symptoms.strip():
                vec = vectorizer.transform([user_symptoms.lower()])
                scores = cosine_similarity(vec, vectors).flatten()
                top3 = scores.argsort()[-3:][::-1]

                for i, idx in enumerate(top3):
                    d = df.iloc[idx]
                    confidence = int(scores[idx] * 100)

                    st.markdown(f"""
                    <div class="card">
                        <h3>#{i+1} {d['name']}</h3>
                        <p><b>Confidence:</b> {confidence}%</p>
                        <p><b>Doctor to Visit:</b> {d['doctor']}</p>
                        <p><b>Precautions:</b> {d['precautions']}</p>
                    </div>
                    """, unsafe_allow_html=True)

    # ---------- DASHBOARD ----------
    with tab2:
        st.subheader("⚡ Quick Health Scan")
        age = st.slider("Age", 18, 80)
        quick_symptom = st.text_input("Enter main symptom")

        if st.button("Quick Scan"):
            if quick_symptom.strip():
                vec = vectorizer.transform([quick_symptom.lower()])
                scores = cosine_similarity(vec, vectors).flatten()
                best = df.iloc[scores.argmax()]

                st.markdown(f"""
                <div class="card">
                    <h2>{best['name']}</h2>
                    <p><b>Doctor:</b> {best['doctor']}</p>
                    <p><b>Action:</b> Visit doctor soon</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("👈 Please login from the sidebar to continue")

st.markdown("---")
st.caption("🏥 MedAI Pro | Educational Demo | © 2026")