import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universal Data Cleaner for downloading. 
    Fills missing values smartly WITHOUT dropping partial rows.
    Keep text as text so humans can read the downloaded CSV.
    """
    df_clean = df.copy()

    # Sirf wahi row/column udayega jo 100% khali hain
    df_clean.dropna(how='all', inplace=True) 
    df_clean.dropna(how='all', axis=1, inplace=True) 

    # Smart Imputation (Filling Nulls)
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                fill_val = df_clean[col].median()
                if pd.isna(fill_val): fill_val = 0
                df_clean[col] = df_clean[col].fillna(fill_val)
            else:
                mode_series = df_clean[col].mode()
                fill_val = mode_series[0] if not mode_series.empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(fill_val)
    
    logger.info("✅ Data completely cleaned for human reading & download.")
    return df_clean

def prepare_for_training(df: pd.DataFrame, drop_columns: list = None) -> pd.DataFrame:
    """
    Prepares data ONLY for Machine Learning.
    Adds Advanced Feature Engineering, drops leakage columns, and encodes text.
    """
    df_ml = df.copy()
    
    # --- 1. DROP LEAKAGE COLUMNS ---
    if drop_columns:
        cols_to_drop = [c for c in drop_columns if c in df_ml.columns]
        df_ml.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"🗑️ Dropped columns for ML training: {cols_to_drop}")

    # --- 2. 🌟 ADVANCED FEATURE ENGINEERING (THE MAGIC) 🌟 ---
    
    # A) Dynamic Age Binning (Teen, Youth, Adult, Senior)
    age_cols = [c for c in df_ml.columns if 'age' in c.lower() and pd.api.types.is_numeric_dtype(df_ml[c])]
    for col in age_cols:
        # Age ko 4 groups mein baant do
        df_ml[f'{col}_Group'] = pd.cut(df_ml[col], bins=[0, 19, 35, 55, 120], labels=['Teen', 'Youth', 'Adult', 'Senior'])
        logger.info(f"✨ Feature Engineered: Created '{col}_Group' from {col}")

    # B) Dynamic Price/Amount Categorization (Low, Medium, High Tier)
    price_keywords = ['price', 'amount', 'usd', 'cost']
    price_cols = [c for c in df_ml.columns if any(k in c.lower() for k in price_keywords) and pd.api.types.is_numeric_dtype(df_ml[c])]
    for col in price_cols:
        # Automatically 3 equal buckets (Low, Medium, High spending)
        df_ml[f'{col}_Tier'] = pd.qcut(df_ml[col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
        logger.info(f"✨ Feature Engineered: Created '{col}_Tier' from {col}")

    # --- 3. LABEL ENCODING (Text to Numbers) ---
    for col in df_ml.columns:
        if df_ml[col].dtype == 'object' or str(df_ml[col].dtype) == 'category':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_ml[col] = df_ml[col].astype(str)
            df_ml[col] = le.fit_transform(df_ml[col])
            
    # --- 4. SAFETY CLEANUP ---
    import numpy as np
    df_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_ml.dropna(inplace=True)
    
    logger.info("🤖 Data successfully Engineered & Encoded for ML Model.")
    return df_ml