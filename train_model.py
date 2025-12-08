import pandas as pd
import numpy as np
import re
import joblib
import json
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- CONFIG ---
DATA_DIR = "db"
MODELS_DIR = "models"
RAW_DATA_PATH = os.path.join(DATA_DIR, "db_computers_2025_raw.csv")
CPU_DB_PATH = os.path.join(DATA_DIR, "db_cpu_raw.csv")
GPU_DB_PATH = os.path.join(DATA_DIR, "db_gpu_raw.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "db_computers_clean.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "price_predictor.joblib")
FEATURE_INFO_PATH = os.path.join(MODELS_DIR, "feature_info.json")
INPUT_OPTIONS_PATH = os.path.join(MODELS_DIR, "input_options.json")

# --- HELPER FUNCTIONS ---
def to_number(x):
    """Extract first number from string with EU-style decimals."""
    if pd.isna(x): return np.nan
    s = str(x)
    match = re.search(r'[\d.,]+', s)
    if not match: return np.nan
    num = match.group(0)
    if ',' in num and '.' in num: num = num.replace('.', '').replace(',', '.')
    elif ',' in num: num = num.replace(',', '.')
    if num == '.' or not num.strip(): return np.nan
    try: return float(num)
    except ValueError: return np.nan

def capacity_to_gb(x):
    """Convert '512 GB' or '1 TB' to numeric GB."""
    if pd.isna(x): return np.nan
    s = str(x).upper()
    value = to_number(s)
    if np.isnan(value): return np.nan
    return value * 1024 if 'TB' in s else value

def normalize_name(x):
    """Normalize CPU/GPU names for matching."""
    if pd.isna(x): return np.nan
    s = str(x).lower()
    s = re.sub(r'®|™|\(r\)|\(tm\)', '', s)
    s = re.sub(r'@.*', '', s)
    s = re.sub(r'\s+cpu$', '', s)
    return re.sub(r'\s+', ' ', s).strip()

# --- MAIN SCRIPT ---
def main():
    print("Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)
    cpu_db = pd.read_csv(CPU_DB_PATH)
    gpu_db = pd.read_csv(GPU_DB_PATH)

    print("Cleaning data...")
    # 1. Extract Numeric Features
    df['price_euros'] = df['Precio_Rango'].apply(to_number)
    df['screen_size_inch'] = df['Pantalla_Tamaño de la pantalla'].apply(to_number)
    df['ram_gb'] = df['RAM_Memoria RAM'].apply(capacity_to_gb)
    df['ssd_gb'] = df['Disco duro_Capacidad de memoria SSD'].apply(capacity_to_gb)
    df['hdd_gb'] = df['Disco duro_Capacidad del disco duro'].apply(capacity_to_gb).fillna(0)
    df['vram_gb'] = df['Gráfica_Memoria gráfica'].apply(capacity_to_gb)
    df['battery_wh'] = df['Alimentación_Vatios-hora'].apply(to_number).fillna(0)
    df['weight_kg'] = df['Medidas y peso_Peso'].apply(to_number)
    df['cpu_base_ghz'] = df['Procesador_Frecuencia de reloj'].apply(to_number)
    df['cpu_turbo_ghz'] = df['Procesador_Frecuencia turbo máx.'].apply(to_number)
    df['cpu_cores'] = df['Procesador_Número de núcleos del procesador'].apply(to_number)
    
    # 2. Categorical Mappings
    df['brand'] = df['Título'].apply(lambda x: x.split()[0] if isinstance(x, str) else 'Other')
    df['prod_type_main'] = df['Tipo de producto']
    
    # 3. Enrich with Benchmarks
    df['cpu_norm'] = df['Procesador_Procesador'].apply(normalize_name)
    cpu_db['cpu_norm'] = cpu_db['CPU Name'].apply(normalize_name)
    # Merge CPU
    cpu_db = cpu_db.drop_duplicates(subset=['cpu_norm'])
    df = df.merge(cpu_db[['cpu_norm', 'CPU Mark (higher is better)', 'Rank (lower is better)', 'CPU Value (higher is better)']], on='cpu_norm', how='left')
    df.rename(columns={'CPU Mark (higher is better)': 'cpu_mark', 'Rank (lower is better)': 'cpu_rank', 'CPU Value (higher is better)': 'cpu_value'}, inplace=True)

    df['gpu_norm'] = df['Gráfica_Tarjeta gráfica'].apply(normalize_name)
    gpu_db['gpu_norm'] = gpu_db['Videocard Name'].apply(normalize_name)
    # Merge GPU
    gpu_db = gpu_db.drop_duplicates(subset=['gpu_norm'])
    df = df.merge(gpu_db[['gpu_norm', 'Passmark G3D Mark (higher is better)', 'Rank (lower is better)', 'Videocard Value (higher is better)']], on='gpu_norm', how='left')
    df.rename(columns={'Passmark G3D Mark (higher is better)': 'gpu_mark', 'Rank (lower is better)': 'gpu_rank', 'Videocard Value (higher is better)': 'gpu_value'}, inplace=True)

    # 4. Handle Missing Values
    numeric_cols = ['screen_size_inch', 'ram_gb', 'ssd_gb', 'vram_gb', 'weight_kg', 
                    'cpu_base_ghz', 'cpu_turbo_ghz', 'cpu_cores', 
                    'cpu_mark', 'gpu_mark']
    
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    df = df.dropna(subset=['price_euros'])
    
    # 5. Feature Engineering
    df['total_storage'] = df['ssd_gb'] + df['hdd_gb']
    if 'proccessor_name' not in df.columns:
        df['proccessor_name'] = df['Procesador_Procesador']
    if 'graphics_name' not in df.columns:
        df['graphics_name'] = df['Gráfica_Tarjeta gráfica']

    print(f"Saving cleaned data to {CLEAN_DATA_PATH}...")
    df.to_csv(CLEAN_DATA_PATH, index=False)
    
    # --- MODEL TRAINING ---
    print("Training model...")
    features = ['screen_size_inch', 'ram_gb', 'ssd_gb', 'vram_gb', 'weight_kg', 
                'cpu_base_ghz', 'cpu_turbo_ghz', 'cpu_cores', 'battery_wh',
                'cpu_mark', 'gpu_mark']
    target = 'price_euros'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model R2: {model.score(X_test, y_test):.4f}")
    
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    
    feature_info = {
        "selected_features": features,
        "feature_stats": {
            col: {
                "median": float(df[col].median()),
                "mean": float(df[col].mean())
            } for col in features
        },
        "encoding_mappings": {},
        "low_card_dummies": {}
    }
    
    input_options = {
        "high_cardinality": {},
        "low_cardinality": {},
        "numeric": features
    }
    
    with open(FEATURE_INFO_PATH, 'w') as f:
        json.dump(feature_info, f, indent=2)
        
    with open(INPUT_OPTIONS_PATH, 'w') as f:
        json.dump(input_options, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
