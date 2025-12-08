import pandas as pd
import numpy as np
import joblib
import json
import os
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import shap

import logging
import sys

# Configure Logging
def setup_logging():
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Constants
DATA_PATH = os.path.join("db", "db_computers_clean.csv")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "price_predictor.joblib")
FEATURE_INFO_PATH = os.path.join(MODELS_DIR, "feature_info.json")
INPUT_OPTIONS_PATH = os.path.join(MODELS_DIR, "input_options.json")

@st.cache_data
def load_data():
    """Loads the raw computer dataset."""
    if os.path.exists(DATA_PATH):
        logger.info(f"Loading data from {DATA_PATH}")
        try:
            df = pd.read_csv(DATA_PATH, low_memory=False)
            
            # Force numeric types for critical columns
            numeric_cols = ['price_euros', 'ram_gb', 'ssd_gb', 'screen_size_inch']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Data loaded successfully: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return None
    logger.error(f"Data file not found at {DATA_PATH}")
    return None

@st.cache_resource
def load_model_resources():
    """Loads the trained model and metadata."""
    logger.info("Loading model resources...")
    try:
        model = joblib.load(MODEL_PATH)
        
        with open(FEATURE_INFO_PATH, 'r') as f:
            feature_info = json.load(f)
            
        with open(INPUT_OPTIONS_PATH, 'r') as f:
            input_options = json.load(f)
            
        logger.info("Model resources loaded successfully")
        return model, feature_info, input_options
    except Exception as e:
        logger.error(f"Error loading model resources: {e}")
        raise e

def preprocess_input(user_input, feature_info):
    """Preprocesses user input dictionary into a model-ready DataFrame."""
    logger.info(f"Preprocessing input: {user_input}")
    # 1. Create DataFrame from input
    df = pd.DataFrame([user_input])
    
    # 2. Impute missing numerical values with median from training data
    for feat, stats in feature_info['feature_stats'].items():
        if feat in df.columns and (df[feat].isnull().any() or df[feat].iloc[0] == 0): 
             df[feat] = df[feat].fillna(stats['median'])

    # 3. Encode Categoricals
    for col, mapping in feature_info['encoding_mappings'].items():
        if col in df.columns:
            val = df[col].iloc[0]
            encoded_val = mapping.get(val, mapping.get("Missing", 0))
            df[col] = encoded_val
            
    # One-Hot Encoding
    if 'low_card_dummies' in feature_info:
        for original_col, details in feature_info['low_card_dummies'].items():
            if original_col in user_input:
                user_val = user_input[original_col]
                for target_col in details['columns']:
                    category = target_col.split('__')[-1]
                    df[target_col] = 1 if user_val == category else 0

    # 4. Ensure all selected features exist and are in order
    final_df = pd.DataFrame()
    for feat in feature_info['selected_features']:
        if feat in df.columns:
            final_df[feat] = df[feat]
        else:
            final_df[feat] = 0 
            
    return final_df

def preprocess_batch(df_batch, feature_info):
    """Preprocesses a batch of data (DataFrame) for SHAP analysis."""
    logger.info(f"Preprocessing batch of {len(df_batch)} rows")
    
    # 1. Handle Missing Numerical Values
    for feat, stats in feature_info['feature_stats'].items():
        if feat in df_batch.columns:
             df_batch[feat] = df_batch[feat].fillna(stats['median'])

    # 2. Encode Categoricals
    for col, mapping in feature_info['encoding_mappings'].items():
        if col in df_batch.columns:
            # Use map to apply encoding, fill missing with 0
            df_batch[col] = df_batch[col].map(mapping).fillna(0)
            
    # 3. One-Hot Encoding (if applicable)
    if 'low_card_dummies' in feature_info:
        for original_col, details in feature_info['low_card_dummies'].items():
            if original_col in df_batch.columns:
                for target_col in details['columns']:
                    category = target_col.split('__')[-1]
                    df_batch[target_col] = (df_batch[original_col] == category).astype(int)

    # 4. Select and Order Features
    final_df = pd.DataFrame()
    for feat in feature_info['selected_features']:
        if feat in df_batch.columns:
            final_df[feat] = df_batch[feat]
        else:
            final_df[feat] = 0
            
    # Ensure numeric types
    final_df = final_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return final_df

def predict_price(model, input_df):
    """Predicts price and returns SHAP values."""
    try:
        prediction = model.predict(input_df)[0]
        logger.info(f"Prediction successful: €{prediction:.2f}")
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        return prediction, explainer, shap_values
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise e

def perform_clustering(df, features, n_clusters=4):
    """Performs K-Means clustering on the dataset."""
    logger.info(f"Running clustering with K={n_clusters} on features {features}")
    try:
        # Drop NaNs for clustering or impute
        data = df[features].dropna()
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Join back to original indices
        df_clustered = df.copy()
        df_clustered.loc[data.index, 'Cluster'] = clusters
        
        # Log sample values to debug visualization
        logger.info(f"Clustered Data Sample:\n{df_clustered[features + ['Cluster']].head()}")
        
        logger.info("Clustering completed")
        return df_clustered
    except Exception as e:
        logger.error(f"Error during clustering: {e}")
        raise e

def find_similar_products(df, user_input_df, features, k=5):
    """Finds k nearest neighbors to the user input."""
    logger.info(f"Finding {k} similar products based on {features}")
    try:
        db_data = df[features].fillna(0) 
        
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(db_data)
        
        distances, indices = nn.kneighbors(user_input_df[features])
        
        results = df.iloc[indices[0]]
        logger.info(f"Found {len(results)} similar products")
        return results
    except Exception as e:
        logger.error(f"Error finding similar products: {e}")
        raise e

def save_feedback(page, rating, comment, filename="feedback_log.csv"):
    """Saves user feedback to a CSV file."""
    logger.info(f"Saving feedback for {page}: {rating}/5")
    feedback_data = {
        "timestamp": pd.Timestamp.now(),
        "page": page,
        "rating": rating,
        "comment": comment
    }
    
    df = pd.DataFrame([feedback_data])
    
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)
