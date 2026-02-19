import streamlit as st
import pandas as pd
import os
import joblib
import json
import requests
import altair as alt  # For beautiful charts
from streamlit_lottie import st_lottie

# Import Backend Modules
from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.train import train_model

# Page Config
st.set_page_config(page_title="Universal MLOps Pipeline", page_icon="🧠", layout="wide")

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# UI Header
col1, col2 = st.columns([1, 4])
with col1:
    lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
    if lottie_ai: st_lottie(lottie_ai, height=150, key="ai")
with col2:
    st.title("🧠 Universal MLOps Pipeline")
    st.write("### Data Agnostic Modeling System")
    st.write("Automated Ingestion • Preprocessing • Training • Inference")

st.markdown("---")

tab1, tab2 = st.tabs(["🚂 **Train Model**", "🔮 **Prediction Interface**"])

# --- TAB 1: TRAIN ---
with tab1:
    st.subheader("Step 1: Ingest & Train")
    uploaded_file = st.file_uploader("Upload Dataset (CSV, XLSX, JSON)", type=["csv", "xlsx", "json"])

    if uploaded_file:
        save_dir = "data/raw"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        
        df = load_data(file_path)
        st.write("📊 Data Preview:")
        st.dataframe(df.head(3))
        
        all_cols = df.columns.tolist()
        target_col = st.selectbox("🎯 Select Target Column to Predict:", all_cols, index=len(all_cols)-1)
        
        if st.button("🚀 Initialize Training Pipeline", type="primary"):
            with st.spinner("Processing Pipeline..."):
                try:
                    # 1. Save Decoder Map (Silent Logic)
                    if df[target_col].dtype == 'object':
                        unique_values = sorted(df[target_col].dropna().unique().tolist())
                        target_map = {i: v for i, v in enumerate(unique_values)}
                        with open("models/target_map.json", "w") as f: json.dump(target_map, f)
                    else:
                        if os.path.exists("models/target_map.json"): os.remove("models/target_map.json")

                    # 2. Process & Train
                    df_clean = preprocess_data(df)
                    os.makedirs("data/processed", exist_ok=True)
                    df_clean.to_csv("data/processed/clean_data.csv", index=False)
                    
                    # Note: Using your updated train.py that returns metrics
                    metrics = train_model("data/processed/clean_data.csv", "models/model.pkl", target_col)
                    
                    st.success(f"✅ Training Complete! Model optimized for target: '{target_col}'")
                    st.balloons()
                    
                    # Show Accuracy (Simple Metric)
                    if "accuracy" in metrics:
                        st.metric("🎯 Accuracy", f"{metrics['accuracy']}%")
                    elif "r2_score" in metrics:
                        st.metric("📈 R2 Score", f"{metrics['r2_score']}")
                    
                except Exception as e:
                    st.error(f"❌ Pipeline Error: {e}")

# --- TAB 2: PREDICT ---
with tab2:
    st.subheader("🔮 Inference Mode")
    model_path = "models/model.pkl"
    meta_path = "models/model_meta.pkl"
    
    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            model = joblib.load(model_path)
            meta = joblib.load(meta_path)
            target_col = meta.get('target_col', 'Unknown')
            
            st.info(f"🤖 Model loaded. Ready to predict: **{target_col}**")
            
            processed_path = "data/processed/clean_data.csv"
            df_schema = pd.read_csv(processed_path)
            feature_cols = [c for c in df_schema.columns if c != target_col]
            
            with st.form("inference_form"):
                input_data = {}
                cols = st.columns(2)
                for i, col in enumerate(feature_cols):
                    with cols[i % 2]:
                        input_data[col] = st.number_input(f"{col}", value=0.0)
                
                if st.form_submit_button("Run Prediction"):
                    input_df = pd.DataFrame([input_data])
                    prediction = model.predict(input_df)[0]
                    
                    # --- DECODER LOGIC (Silent) ---
                    # Ye check karega ki kya hum '0' ko 'Clothing' mein badal sakte hain?
                    final_output = prediction
                    map_path = "models/target_map.json"
                    target_map = {}
                    
                    if os.path.exists(map_path):
                        with open(map_path, "r") as f: target_map = json.load(f)
                        # Agar prediction match ho jaye map se to text lelo
                        if str(int(prediction)) in target_map:
                            final_output = target_map[str(int(prediction))]

                    st.markdown("---")
                    
                    # 1. Simple Result (Tera Favourite Format)
                    st.success(f"💡 Prediction Result: **{final_output}**")
                    
                    # 2. The Cool Chart (Only if Probability exists)
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_df)[0]
                        
                        # Labels set karna
                        if target_map:
                            labels = [target_map[str(i)] for i in range(len(proba))]
                        else:
                            labels = [f"Class {i}" for i in range(len(proba))]
                            
                        # Data for Chart
                        chart_data = pd.DataFrame({
                            "Category": labels,
                            "Confidence": proba
                        })

                        # --- NEW: DONUT CHART (Interactive & Cool) ---
                        base = alt.Chart(chart_data).encode(
                            theta=alt.Theta("Confidence", stack=True)
                        )
                        pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
                            color=alt.Color("Category"),
                            order=alt.Order("Confidence", sort="descending"),
                            tooltip=["Category", alt.Tooltip("Confidence", format=".1%")]
                        )
                        text = base.mark_text(radius=140).encode(
                            text=alt.Text("Confidence", format=".1%"),
                            order=alt.Order("Confidence", sort="descending"),
                            color=alt.value("white")
                        )
                        
                        st.write("### 📊 Confidence Analysis")
                        st.altair_chart(pie + text, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("⚠️ No trained model found. Please train a model in the 'Train' tab first.")