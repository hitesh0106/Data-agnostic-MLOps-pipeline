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
from src.preprocessing.preprocess import clean_raw_data, prepare_for_training # NEW IMPORTS
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

# --- GLOBAL UPLOAD SECTION ---
st.subheader("📁 Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload Dataset (CSV, XLSX, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file:
    save_dir = "data/raw"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    
    # Load raw data globally so all tabs can use it
    df_raw = load_data(file_path)
    
    # Universal Cleaning (Runs instantly so it's ready for Tab 1 & 2)
    df_cleaned = clean_raw_data(df_raw)

    st.markdown("---")
    
    # --- THE 3 NEW TABS ---
    tab1, tab2, tab3 = st.tabs(["🧹 **Clean & Download**", "⚙️ **Train Model**", "🔮 **Prediction Interface**"])

    # ==========================================
    # TAB 1: DATA CLEANER
    # ==========================================
    with tab1:
        st.subheader("🧹 Universal Data Cleaner")
        st.write("Missing values filled intelligently. Zero rows deleted. Ready for analysis!")
        
        st.write("📊 Cleaned Data Preview:")
        st.dataframe(df_cleaned.head(50))
        
        # Download Button
        csv_data = df_cleaned.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaned Data (CSV)",
            data=csv_data,
            file_name="universal_cleaned_data.csv",
            mime="text/csv",
            type="primary"
        )

    # ==========================================
    # TAB 2: TRAIN MODEL
    # ==========================================
    with tab2:
        st.subheader("⚙️ Step 2: Train AI Model")
        
        all_cols = df_cleaned.columns.tolist()
        
        # UI for Target and Drop Columns
        col_t, col_d = st.columns(2)
        with col_t:
            target_col = st.selectbox("🎯 Select Target Column to Predict:", all_cols, index=len(all_cols)-1)
        with col_d:
            available_drop_cols = [c for c in all_cols if c != target_col]
            drop_cols = st.multiselect("🗑️ Drop Leakage Columns (Promo Codes, IDs):", available_drop_cols)
        
        if st.button("🚀 Initialize Training Pipeline", type="primary"):
            with st.spinner("Processing Pipeline & Training Model..."):
                try:
                    # 1. Save Decoder Map (Tera original silent logic, untouched!)
                    if df_cleaned[target_col].dtype == 'object':
                        unique_values = sorted(df_cleaned[target_col].dropna().unique().tolist())
                        target_map = {i: v for i, v in enumerate(unique_values)}
                        os.makedirs("models", exist_ok=True)
                        with open("models/target_map.json", "w") as f: json.dump(target_map, f)
                    else:
                        if os.path.exists("models/target_map.json"): os.remove("models/target_map.json")

                    # 2. Prepare for ML (Encode + Drop Columns)
                    df_ml = prepare_for_training(df_cleaned, drop_columns=drop_cols)
                    
                    # Save exact schema for inference tab
                    os.makedirs("data/processed", exist_ok=True)
                    df_ml.to_csv("data/processed/clean_data.csv", index=False)
                    
                    # 3. Train Model
                    metrics = train_model("data/processed/clean_data.csv", "models/model.pkl", target_col)
                    
                    st.success(f"✅ Training Complete! Model optimized for target: '{target_col}'")
                    st.balloons() # Tera favourite!
                    
                    # Show Accuracy
                    if "accuracy" in metrics:
                        st.metric("🎯 Accuracy", f"{metrics['accuracy']}%")
                    elif "r2_score" in metrics:
                        st.metric("📈 R2 Score", f"{metrics['r2_score']}")
                    
                except Exception as e:
                    st.error(f"❌ Pipeline Error: {e}")

    # ==========================================
    # TAB 3: PREDICTION INTERFACE (100% UNTOUCHED)
    # ==========================================
    with tab3:
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
                        final_output = prediction
                        map_path = "models/target_map.json"
                        target_map = {}
                        
                        if os.path.exists(map_path):
                            with open(map_path, "r") as f: target_map = json.load(f)
                            if str(int(prediction)) in target_map:
                                final_output = target_map[str(int(prediction))]

                        st.markdown("---")
                        
                        # 1. Simple Result
                        st.success(f"💡 Prediction Result: **{final_output}**")
                        
                        # 2. The Cool Chart
                        if hasattr(model, "predict_proba"):
                            proba = model.predict_proba(input_df)[0]
                            
                            if target_map:
                                labels = [target_map[str(i)] for i in range(len(proba))]
                            else:
                                labels = [f"Class {i}" for i in range(len(proba))]
                                
                            chart_data = pd.DataFrame({
                                "Category": labels,
                                "Confidence": proba
                            })

                            # DONUT CHART 
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
            st.warning("⚠️ No trained model found. Please train a model in the 'Train Model' tab first.")
else:
    st.info("👈 Upload your dataset to get started!")