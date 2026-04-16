import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Missing values ko smart tarike se fill karo
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
    return df

def prepare_for_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 🌟 GOD-MODE: Convert literally ANY text, email, or ID into numbers
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            # Convert everything to string first to avoid mixed type crashes
            df[col] = le.fit_transform(df[col].astype(str))
    return df