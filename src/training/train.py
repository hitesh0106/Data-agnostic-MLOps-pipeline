import pandas as pd
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, model_save_path: str, target_column: str) -> dict:
    """
    Trains a model. Now supports up to 50 categories for Classification.
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        # X and y
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Detect Problem Type
        unique_values = y.nunique()
        metrics = {}
        
        # --- CHANGE IS HERE: 10 -> 50 ---
        # Ab agar 50 colors bhi honge, tab bhi wo Classification hi karega
        if unique_values <= 50:
            problem_type = "Classification"
            model = LogisticRegression(max_iter=3000) # Iterations bhi badha di taaki complex data handle ho
            model.fit(X_train, y_train)
            
            # Accuracy
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            metrics = {"accuracy": round(acc * 100, 2)}
            
        else:
            problem_type = "Regression"
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # R2 Score
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            metrics = {"r2_score": round(r2, 2)}

        # Save Logic
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        
        meta_data = {
            "target_col": target_column, 
            "problem_type": problem_type,
            "metrics": metrics
        }
        joblib.dump(meta_data, model_save_path.replace(".pkl", "_meta.pkl"))
        
        logger.info(f"💾 Model Saved. Type: {problem_type}")
        return metrics

    except Exception as e:
        logger.error(f"❌ Training Error: {e}")
        raise e