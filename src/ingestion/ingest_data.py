import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data and REMOVES USELESS COLUMNS (IDs, Names).
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load Data
        if file_path.endswith(".csv"):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

        # --- NEW: DROP USELESS COLUMNS ---
        # Ye list mein wo naam daal jo model ko confuse karte hain
        useless_cols = ["Customer ID", "PassengerId", "Name", "Ticket", "Date", "User ID"]
        
        # Agar column data mein hai, to uda do
        for col in useless_cols:
            # Case insensitive check (agar 'customer id' likha ho to bhi pakad le)
            matching_cols = [c for c in df.columns if c.lower() == col.lower()]
            if matching_cols:
                df = df.drop(columns=matching_cols)
                logger.info(f"🗑️ Dropped useless column: {matching_cols}")
        # ---------------------------------

        logger.info(f"✅ Data Loaded! Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise e