import streamlit as st
import pandas as pd
import os
import joblib
import json
import requests
import altair as alt
import shap  
import numpy as np
from streamlit_lottie import st_lottie

# Page Config
st.set_page_config(page_title="Universal MLOps Pipeline", page_icon="🧠", layout="wide")

# Import Backend Modules 
from src.ingestion.ingest_data import load_data
from src.preprocessing.preprocess import clean_raw_data, prepare_for_training
from src.training.train import train_model

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
    st.write("Automated Ingestion • Preprocessing • Training • Inference • Explainability")

st.markdown("---")

# --- GLOBAL UPLOAD SECTION ---
st.subheader("📁 Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload Dataset (CSV, XLSX, JSON)", type=["csv", "xlsx", "json"])

if uploaded_file:
    save_dir = "data/raw"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    
    # Load raw data globally
    df_raw = load_data(file_path)
    
    # Universal Cleaning
    df_cleaned = clean_raw_data(df_raw)

    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🧹 **Clean & Download**", "⚙️ **Train Model**", "🔮 **Prediction Interface**"])

    # ==========================================
    # TAB 1: DATA CLEANER
    # ==========================================
    with tab1:
        st.subheader("🧹 Universal Data Cleaner")
        st.write("Missing values filled intelligently. Zero rows deleted. Ready for analysis!")
        st.dataframe(df_cleaned.head(50))
        
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
        
        target_col = st.selectbox("🎯 Select Target Column to Predict:", all_cols, index=len(all_cols)-1)
        
        if st.button("🚀 Initialize Training Pipeline", type="primary"):
            with st.spinner("Processing Pipeline & Training Model..."):
                try:
                    if df_cleaned[target_col].dtype == 'object':
                        unique_values = sorted(df_cleaned[target_col].dropna().unique().tolist())
                        target_map = {i: v for i, v in enumerate(unique_values)}
                        os.makedirs("models", exist_ok=True)
                        with open("models/target_map.json", "w") as f: json.dump(target_map, f)
                    else:
                        if os.path.exists("models/target_map.json"): os.remove("models/target_map.json")

                    df_ml = prepare_for_training(df_cleaned) 
                    os.makedirs("data/processed", exist_ok=True)
                    df_ml.to_csv("data/processed/clean_data.csv", index=False)
                    
                    metrics = train_model("data/processed/clean_data.csv", "models/model.pkl", target_col)
                    
                    st.success(f"✅ Training Complete! Model optimized for target: '{target_col}'")
                    st.balloons()
                    
                    if "accuracy" in metrics:
                        st.metric("🎯 Accuracy", f"{metrics['accuracy']}%")
                    elif "r2_score" in metrics:
                        st.metric("📈 R2 Score", f"{metrics['r2_score']}")
                    
                except Exception as e:
                    st.error(f"❌ Pipeline Error: {e}")

    # ==========================================
    # TAB 3: PREDICTION & XAI (SIDE-BY-SIDE LAYOUT)
    # ==========================================
    with tab3:
        st.subheader("🔮 Inference & Explainability")
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
                            # ✨ YAHAN CHANGE KIYA HAI: step=1.0 add kar diya ✨
                            input_data[col] = st.number_input(f"{col}", value=0.0, step=1.0)
                    
                    if st.form_submit_button("Run Prediction"):
                        input_df = pd.DataFrame([input_data])
                        prediction = model.predict(input_df)[0]
                        
                        final_output = prediction
                        map_path = "models/target_map.json"
                        target_map = {}
                        
                        if os.path.exists(map_path):
                            with open(map_path, "r") as f: target_map = json.load(f)
                            if str(int(prediction)) in target_map:
                                final_output = target_map[str(int(prediction))]

                        st.markdown("---")
                        
                        # 1. PREDICTION RESULT
                        st.success(f"💡 Prediction Result: **{final_output}**")
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        col_xai, col_donut = st.columns(2)
                        xai_vals = None 
                        
                        # 2. XAI (EXPLAINABLE AI) - LEFT COLUMN
                        with col_xai:
                            st.write("### 🧠 AI X-Ray (Why?)")
                            with st.spinner("Scanning..."):
                                try:
                                    model_name = type(model).__name__
                                    
                                    if "Forest" in model_name or "Boosting" in model_name:
                                        explainer = shap.TreeExplainer(model)
                                        shap_vals = explainer.shap_values(input_df)
                                        
                                        if isinstance(shap_vals, list):
                                            pred_idx = int(model.predict(input_df)[0])
                                            vals = shap_vals[pred_idx][0]
                                        else:
                                            vals = shap_vals[0]
                                            if len(vals.shape) > 1:
                                                vals = vals[:, 0]
                                    else:
                                        background_data = df_schema.drop(columns=[target_col]).fillna(0)
                                        explainer = shap.LinearExplainer(model, background_data)
                                        vals = explainer.shap_values(input_df)[0]

                                    xai_vals = vals 
                                    features = input_df.columns
                                    
                                    df_shap = pd.DataFrame({'Feature': features, 'Impact': vals})
                                    df_shap['Abs_Impact'] = df_shap['Impact'].abs()
                                    df_shap = df_shap.sort_values(by='Abs_Impact', ascending=False).head(5) 
                                    df_shap['Color'] = np.where(df_shap['Impact'] > 0, '#ff0051', '#008bfb') 
                                    
                                    bar_chart = alt.Chart(df_shap).mark_bar().encode(
                                        x=alt.X('Impact:Q', title='Impact on Prediction'),
                                        y=alt.Y('Feature:N', sort='-x', title=''),
                                        color=alt.Color('Color:N', scale=None),
                                        tooltip=['Feature', 'Impact']
                                    ).properties(height=250)
                                    
                                    st.altair_chart(bar_chart, use_container_width=True)
                                    st.caption("🟥 **Red:** Pushed prediction higher. | 🟦 **Blue:** Pushed prediction lower.")
                                except Exception as e:
                                    st.warning(f"⚠️ XAI not supported for {type(model).__name__}. ({e})")
                                    
                        # 3. DONUT CHART OR FEATURE INFLUENCE - RIGHT COLUMN
                        with col_donut:
                            if hasattr(model, "predict_proba"):
                                st.write("### 📊 Confidence Analysis")
                                proba = model.predict_proba(input_df)[0]
                                labels = [target_map[str(i)] for i in range(len(proba))] if target_map else [f"Class {i}" for i in range(len(proba))]
                                    
                                chart_data = pd.DataFrame({"Category": labels, "Confidence": proba})

                                base = alt.Chart(chart_data).encode(theta=alt.Theta("Confidence", stack=True))
                                pie = base.mark_arc(outerRadius=100, innerRadius=60).encode(
                                    color=alt.Color("Category"),
                                    order=alt.Order("Confidence", sort="descending"),
                                    tooltip=["Category", alt.Tooltip("Confidence", format=".1%")]
                                )
                                text = base.mark_text(radius=120).encode(
                                    text=alt.Text("Confidence", format=".1%"),
                                    order=alt.Order("Confidence", sort="descending"),
                                    color=alt.value("white")
                                )
                                
                                st.altair_chart(pie + text, use_container_width=True)
                                
                            else:
                                st.write("### ⚖️ Feature Influence")
                                if xai_vals is not None:
                                    pos_force = sum([v for v in xai_vals if v > 0])
                                    neg_force = sum([abs(v) for v in xai_vals if v < 0])
                                    
                                    if pos_force == 0 and neg_force == 0:
                                        st.info("Prediction is at baseline. No major influencing features.")
                                    else:
                                        force_data = pd.DataFrame({
                                            "Impact": ["Increasing Value ⬆️", "Decreasing Value ⬇️"],
                                            "Magnitude": [pos_force, neg_force]
                                        })
                                        
                                        base = alt.Chart(force_data).encode(theta=alt.Theta("Magnitude", stack=True))
                                        pie = base.mark_arc(outerRadius=100, innerRadius=60).encode(
                                            color=alt.Color("Impact", scale=alt.Scale(domain=["Increasing Value ⬆️", "Decreasing Value ⬇️"], range=["#ff0051", "#008bfb"])),
                                            tooltip=["Impact", "Magnitude"]
                                        )
                                        
                                        st.altair_chart(pie, use_container_width=True)
                                        st.caption("Overall balance of features pushing the prediction UP 🟥 vs DOWN 🟦.")
                                else:
                                    st.info("Prediction insights are currently being generated.")

            except Exception as e:
                st.error(f"Error loading model: {e}")
        else:
            st.warning("⚠️ No trained model found. Please train a model in the 'Train Model' tab first.")
else:
    st.info("👈 Upload your dataset to get started!")