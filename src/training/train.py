import pandas as pd
import joblib
import os
import logging
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, model_save_path: str, target_column: str) -> dict:
    """
    AutoML Engine: Races multiple models, tunes hyperparameters, and saves the absolute best one.
    """
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        unique_values = y.nunique()
        best_model = None
        best_score = -float('inf')
        best_model_name = ""
        metrics = {}
        
        # ==========================================
        # 🎯 CLASSIFICATION RACE (If unique values <= 50)
        # ==========================================
        if unique_values <= 50:
            problem_type = "Classification"
            logger.info("🚦 Starting Classification AutoML Race...")
            
            models = {
                "RandomForest": (RandomForestClassifier(random_state=42), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                }),
                "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 0.2]
                }),
                "LogisticRegression": (LogisticRegression(max_iter=2000), {
                    'C': [0.1, 1.0, 10.0]
                })
            }

            for name, (model, params) in models.items():
                logger.info(f"🔧 Tuning & Training {name}...")
                # RandomizedSearchCV best setting dhundhega
                search = RandomizedSearchCV(model, params, n_iter=5, cv=3, random_state=42, n_jobs=-1)
                search.fit(X_train, y_train)
                
                # Test the best version of this model
                y_pred = search.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                
                logger.info(f"🏁 {name} finished with Accuracy: {round(score * 100, 2)}%")
                
                if score > best_score:
                    best_score = score
                    best_model = search.best_estimator_
                    best_model_name = name
                    metrics = {"accuracy": round(best_score * 100, 2), "winning_model": best_model_name}

        # ==========================================
        # 📈 REGRESSION RACE (If unique values > 50)
        # ==========================================
        else:
            problem_type = "Regression"
            logger.info("🚦 Starting Regression AutoML Race...")
            
            models = {
                "RandomForest_Reg": (RandomForestRegressor(random_state=42), {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                }),
                "GradientBoosting_Reg": (GradientBoostingRegressor(random_state=42), {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 0.2]
                }),
                "LinearRegression": (LinearRegression(), {}) # Linear Reg has no major params to tune
            }

            for name, (model, params) in models.items():
                logger.info(f"🔧 Tuning & Training {name}...")
                if params:
                    search = RandomizedSearchCV(model, params, n_iter=5, cv=3, random_state=42, n_jobs=-1)
                    search.fit(X_train, y_train)
                    current_best_model = search.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    current_best_model = model
                
                # Test the model
                y_pred = current_best_model.predict(X_test)
                score = r2_score(y_test, y_pred)
                
                logger.info(f"🏁 {name} finished with R2 Score: {round(score, 4)}")
                
                if score > best_score:
                    best_score = score
                    best_model = current_best_model
                    best_model_name = name
                    metrics = {"r2_score": round(best_score, 4), "winning_model": best_model_name}

        # ==========================================
        # 🏆 SAVE THE CHAMPION
        # ==========================================
        logger.info(f"🏆 CHAMPION MODEL: {best_model_name} with score {best_score}")
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(best_model, model_save_path)
        
        meta_data = {
            "target_col": target_column, 
            "problem_type": problem_type,
            "metrics": metrics,
            "model_algorithm": best_model_name
        }
        joblib.dump(meta_data, model_save_path.replace(".pkl", "_meta.pkl"))
        
        return metrics

    except Exception as e:
        logger.error(f"❌ AutoML Training Error: {e}")
        raise e