import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# 1. Page Configuration
st.set_page_config(page_title="NASA Engine RUL Predictor", page_icon="✈️", layout="wide")

# 2. Main Title and Subtitle
st.title("✈️ Predictive Maintenance: Engine RUL Prediction")
st.markdown("### Predicting the Remaining Useful Life (RUL) of Aircraft Engines Based on Sensor Readings")
st.markdown("---")

# 3. Sidebar Configuration
st.sidebar.header("⚙️ Model Settings")
dataset_choice = st.sidebar.selectbox(
    "Select Dataset (Operating Conditions):",
    ["FD001 (1 Fault / 1 Operating Condition)", 
     "FD002 (1 Fault / 6 Operating Conditions)", 
     "FD003 (2 Faults / 1 Operating Condition)", 
     "FD004 (2 Faults / 6 Operating Conditions)"]
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload Test Data File (.txt)", type=['txt'])

# 4. Function to Load Models
@st.cache_resource
def load_models(choice):
    try:
        if "FD001" in choice:
            model = joblib.load('model.pkl')
            scaler = joblib.load('scaler.pkl')
            return model, scaler, None
            
        elif "FD002" in choice:
            model = joblib.load('xgb_model_fd002.pkl')
            scaler = joblib.load('scaler_fd002.pkl')
            kmeans = joblib.load('kmeans_fd002.pkl') 
            return model, scaler, kmeans
            
        elif "FD003" in choice:
            model = joblib.load('model_FD003.pkl')
            scaler = joblib.load('scaler_FD003.pkl')
            return model, scaler, None
            
        elif "FD004" in choice:
            model = joblib.load('xgb_model_fd004.pkl')
            scaler = joblib.load('scaler_fd004.pkl')
            kmeans = joblib.load('kmeans_fd004.pkl')
            return model, scaler, kmeans
            
    except Exception as e:
        return None, None, None
        
    return None, None, None

# 5. Execution when a file is uploaded
if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    
    # Read the uploaded data
    test_data = pd.read_csv(uploaded_file, sep=r'\s+', header=None)
    
    # Prepare column names
    index_names = ['unit_number', 'time_in_cycles']
    setting_names = ['setting1', 'setting2', 'setting3']
    sensor_names = ['sensor{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    features = setting_names + sensor_names
    
    # Clean and structure uploaded data
    test_data.dropna(axis=1, how='all', inplace=True)
    test_data.columns = col_names
    
    st.write("📊 **Quick Glance at the Uploaded Data:**")
    st.dataframe(test_data.head())
    
    st.markdown("---")
    
    # ==========================================
    # Engine Selection Dropdown
    # ==========================================
    unique_engines = test_data['unit_number'].unique()
    selected_engine = st.selectbox("🔍 Select the Engine Number to Inspect:", unique_engines)
    
    if st.button("🚀 Predict RUL"):
        
        with st.spinner(f'Analyzing sensor readings for Engine #{selected_engine}...'):
            try:
                # Load Models
                model, scaler, kmeans = load_models(dataset_choice)
                if model is None:
                    st.error("⚠️ Model files (.pkl) for this dataset are missing! Ensure they are saved in the same directory.")
                    st.stop()
                
                current_features = features
                
                # Determine Dynamic Window
                if "FD001" in dataset_choice: window = 15
                elif "FD002" in dataset_choice: window = 5
                elif "FD003" in dataset_choice: window = 30
                elif "FD004" in dataset_choice: window = 5
                
                # Feature Engineering (Rolling Mean & Std)
                for col in current_features:
                    test_data[col + '_mean'] = test_data.groupby('unit_number')[col].transform(lambda x: x.rolling(window).mean())
                    test_data[col + '_std'] = test_data.groupby('unit_number')[col].transform(lambda x: x.rolling(window).std())
                test_data = test_data.bfill()
                
                # Extract Selected Engine Data
                engine_data = test_data[test_data['unit_number'] == selected_engine].tail(1).copy()
                
                # Scaling
                scaler_cols = list(scaler.feature_names_in_)
                for col in scaler_cols:
                    if col not in engine_data.columns:
                        engine_data[col] = 0
                        
                engine_data[scaler_cols] = scaler.transform(engine_data[scaler_cols])
                
                # Clustering (K-Means)
                if kmeans is not None:
                    kmeans_cols = list(kmeans.feature_names_in_)
                    engine_data['cluster'] = kmeans.predict(engine_data[kmeans_cols])
                    engine_data = pd.get_dummies(engine_data, columns=['cluster'], prefix='Condition')
                
                # =========================================================
                # 🔥 التعديل هنا: Final Feature Alignment (عشان مشكلة Random Forest)
                # =========================================================
                try:
                    # لو الموديل XGBoost وحافظ الأسماء
                    model_cols = list(model.feature_names_in_)
                except AttributeError:
                    # لو الموديل Random Forest ومش حافظ الأسماء، هناخدها من السكالر
                    model_cols = list(scaler.feature_names_in_)
                    if kmeans is not None:
                        cluster_cols = [c for c in engine_data.columns if c.startswith('Condition_')]
                        model_cols.extend(cluster_cols)
                
                X_final = engine_data.reindex(columns=model_cols, fill_value=0)
                X_final = X_final[model_cols]
                # =========================================================
                
                # RUL Prediction
                real_prediction = int(model.predict(X_final)[0])
                
                st.markdown("---")
                st.subheader(f"🎯 Final Result for Engine #{int(selected_engine)}:")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Engine Number", f"Engine #{int(selected_engine)}")
                
                if real_prediction < 30:
                    col2.metric("Remaining Useful Life (RUL)", f"{real_prediction} Cycles", delta="- CRITICAL! Maintenance Required", delta_color="normal")
                else:
                    col2.metric("Remaining Useful Life (RUL)", f"{real_prediction} Cycles", delta="+ Healthy / Safe Condition", delta_color="normal")
                
                st.balloons()
                st.markdown("---")
                st.markdown(f"### 📈 Sensor Degradation History for Engine #{int(selected_engine)}")
                
                # بنجيب تاريخ المحرك ده من أول ما طار لحد آخر لحظة
                engine_history = test_data[test_data['unit_number'] == selected_engine]
                
                chart_data = engine_history[['time_in_cycles', 'sensor13', 'sensor14']].set_index('time_in_cycles')
                
                # رسم بياني تفاعلي (الدكتور يقدر يزوم فيه)
                st.line_chart(chart_data)
                
            except Exception as e:
                st.error(f"⚠️ An error occurred during analysis: {e}")
                st.info("Make sure you uploaded the correct test file corresponding to the selected model.")
else:
    st.info("👈 Please upload the Test Data file from the sidebar to begin.")