import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Universal Data Loader: Reads CSV, Excel, and JSON files seamlessly.
    """
    logger.info(f"Loading data from: {file_path}")
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
            
        elif file_path.endswith('.json'):
            # First trick: Seedha padhne ki koshish (If JSON is simple and flat)
            try:
                df = pd.read_json(file_path)
            except Exception as e:
                logger.warning("Normal JSON read failed. It might be nested. Trying Smart Flattening...")
                # Second trick: Agar JSON nested hai, toh usko flatten (seedha) kar do
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Agar data ek dictionary ke andar band hai, toh json_normalize usko proper table bana dega
                df = pd.json_normalize(data)
                
        else:
            raise ValueError("Bhai, sirf CSV, Excel ya JSON format hi allow hai!")
            
        logger.info(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise e