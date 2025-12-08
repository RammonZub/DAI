import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import app_utils

# Page Config
st.set_page_config(page_title="ML Marketplace", page_icon="💻", layout="wide")

# Load Data & Models
try:
    df = app_utils.load_data()
    model, feature_info, input_options = app_utils.load_model_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/laptop--v1.png", width=50)
st.sidebar.title("ML Marketplace")
st.sidebar.info("This app predicts computer prices based on hardware specifications using machine learning.")

# Main Title
st.title("ML Marketplace: Computer Configuration")
st.markdown("Create your perfect computer configuration and get an estimated price range. Use the smart presets for quick setups or customize each component.")

# Tabs
tab1, tab2, tab3 = st.tabs(["💻 Price Predictor & Analysis", "🔍 Browse Computers", "📊 Market Segmentation"])

# --- TAB 1: PRICE PREDICTOR ---
with tab1:
    st.subheader("Computer Configuration")
    st.markdown("Configure your computer specifications and get a predicted price range.")
    
    # Initialize session state
    default_values = {
        "screen_size_inch": 15.6, "ram_gb": 16, "ssd_gb": 512, "cpu_cores": 8,
        "cpu_base_ghz": 2.5, "cpu_turbo_ghz": 4.5, "vram_gb": 4.0, "battery_wh": 50.0
    }
    for key, val in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Quick Profiles
    st.markdown("### 🚀 Quick Configuration Profiles")
    st.markdown("Select a preset configuration based on your needs:")
    
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    def update_profile(profile_dict):
        for k, v in profile_dict.items():
            st.session_state[k] = v
    
    if col_p1.button("💰 Budget Friendly"):
        update_profile({"ram_gb": 8, "ssd_gb": 256, "cpu_cores": 4, "vram_gb": 0.0})
        st.success("Applied 'Budget Friendly' configuration!")
    if col_p2.button("⚖️ Balanced Performer"):
        update_profile({"ram_gb": 16, "ssd_gb": 512, "cpu_cores": 8, "vram_gb": 4.0})
        st.success("Applied 'Balanced Performer' configuration!")
    if col_p3.button("🎮 Gaming Powerhouse"):
        update_profile({"ram_gb": 32, "ssd_gb": 1000, "cpu_cores": 16, "vram_gb": 12.0})
        st.success("Applied 'Gaming Powerhouse' configuration!")
    if col_p4.button("💼 Ultra-Portable Creator"):
        update_profile({"ram_gb": 64, "ssd_gb": 2000, "cpu_cores": 12, "vram_gb": 8.0, "screen_size_inch": 14.0})
        st.success("Applied 'Ultra-Portable Creator' configuration!")

    # Input Form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        user_input = {}
        
        with col1:
            st.markdown("#### ⚙️ Processor & Memory")
            user_input['cpu_cores'] = st.number_input("CPU Cores", 2, 64, key="cpu_cores")
            user_input['cpu_base_ghz'] = st.number_input("CPU Base GHz", 0.0, 6.0, key="cpu_base_ghz")
            user_input['cpu_turbo_ghz'] = st.number_input("CPU Turbo GHz", 0.0, 7.0, key="cpu_turbo_ghz")
            user_input['ram_gb'] = st.number_input("RAM (GB)", 4, 128, key="ram_gb")

        with col2:
            st.markdown("#### 💾 Storage & Graphics")
            user_input['ssd_gb'] = st.number_input("SSD (GB)", 0, 8000, key="ssd_gb")
            user_input['vram_gb'] = st.number_input("VRAM (GB)", 0.0, 48.0, key="vram_gb")
            
        with col3:
            st.markdown("#### 🖥️ Display & Battery")
            user_input['screen_size_inch'] = st.number_input("Screen Size (inch)", 10.0, 20.0, key="screen_size_inch")
            user_input['battery_wh'] = st.number_input("Battery (Wh)", 0.0, 100.0, key="battery_wh")
            
            # Categorical Inputs
            for feat, options in input_options['high_cardinality'].items():
                if feat in feature_info['selected_features']:
                    user_input[feat] = st.selectbox(feat, options, key=f"input_{feat}")
            for feat, options in input_options['low_cardinality'].items():
                user_input[feat] = st.selectbox(feat, options, key=f"input_{feat}")

        # Hidden defaults
        user_input['height'] = 20.0
        user_input['width'] = 30.0
        user_input['depth'] = 20.0
        user_input['weight_kg'] = 2.0
        user_input['offers_num'] = 1
        user_input['brightness_cd'] = 300
        
        submitted = st.form_submit_button("Predict Price")
        
    if submitted:
        input_df = app_utils.preprocess_input(user_input, feature_info)
        price, explainer, shap_values = app_utils.predict_price(model, input_df)
        
        st.markdown("### Price Estimation")
        st.markdown(f"Predicted Price Range: **€{price*0.9:.0f} - €{price*1.1:.0f}**")
        st.metric("Exact Prediction", f"€{price:.2f}")
        
        st.subheader("Why this price?")
        fig, ax = plt.subplots()
        shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                             base_values=explainer.expected_value[0], 
                                             data=input_df.iloc[0], 
                                             feature_names=input_df.columns))
        st.pyplot(fig)

# --- TAB 2: BROWSE COMPUTERS (DEAL FINDER) ---
with tab2:
    st.subheader("Find Similar Computers")
    
    col1, col2 = st.columns(2)
    with col1:
        search_ram = st.slider("RAM (GB)", 4, 64, 16, key="search_ram")
        search_ssd = st.slider("SSD (GB)", 256, 2000, 512, key="search_ssd")
    with col2:
        search_screen = st.slider("Screen Size", 13.0, 17.0, 15.6, key="search_screen")
        search_price = st.slider("Max Price (€)", 500, 5000, 1500, key="search_price")
        
    if st.button("Find Deals"):
        search_input = pd.DataFrame([{
            'ram_gb': search_ram, 'ssd_gb': search_ssd, 
            'screen_size_inch': search_screen, 'price_euros': search_price
        }])
        sim_features = ['ram_gb', 'ssd_gb', 'screen_size_inch', 'price_euros']
        similar_products = app_utils.find_similar_products(df, search_input, sim_features, k=5)
        
        st.dataframe(similar_products[['brand', 'proccessor_name', 'ram_gb', 'ssd_gb', 'price_euros']])

# --- TAB 3: MARKET SEGMENTATION ---
with tab3:
    st.subheader("Market Segmentation")
    
    cluster_features = ['price_euros', 'ram_gb', 'ssd_gb', 'screen_size_inch']
    n_clusters = st.slider("Number of Clusters", 2, 8, 4)
    
    if st.button("Run Clustering"):
        clustered_df = app_utils.perform_clustering(df, cluster_features, n_clusters)
        
        # Visualization
        plot_df = clustered_df.dropna(subset=['Cluster'])[['price_euros', 'ram_gb', 'Cluster']].copy()
        
        # Explicitly enforce types on the subset
        plot_df['price_euros'] = pd.to_numeric(plot_df['price_euros'], errors='coerce')
        plot_df['ram_gb'] = pd.to_numeric(plot_df['ram_gb'], errors='coerce')
        plot_df['Cluster'] = plot_df['Cluster'].astype(int).astype(str)
        plot_df = plot_df.dropna()
        
        st.markdown(f"### Market Segments (K={n_clusters})")
        st.markdown("Visualizing the identified market segments based on Price and RAM.")
        
        st.scatter_chart(
            plot_df,
            x='price_euros',
            y='ram_gb',
            color='Cluster',
            height=500,
            use_container_width=True
        )
        
        # Profiles
        st.subheader("Cluster Profiles")
        avg_stats = clustered_df.groupby("Cluster")[cluster_features].mean().reset_index()
        st.dataframe(avg_stats.style.format({
            'price_euros': '{:.2f}',
            'ram_gb': '{:.1f}',
            'ssd_gb': '{:.0f}',
            'screen_size_inch': '{:.1f}'
        }))
