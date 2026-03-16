import streamlit as st
import pandas as pd
import joblib

# 1. إعدادات وشكل الصفحة
st.set_page_config(page_title="NASA Engine RUL Predictor", page_icon="✈️", layout="wide")
st.title("✈️ توقع العمر المتبقي لمحركات الطائرات (NASA CMAPSS)")
st.write("هذا التطبيق الذكي يتوقع متى سينهار محرك الطائرة بناءً على قراءات الحساسات وظروف التشغيل.")

# 2. القائمة الجانبية (عشان نختار الداتا)
st.sidebar.header("⚙️ إعدادات الرحلة")
condition = st.sidebar.selectbox(
    "اختر ظروف تشغيل الطيارة:",
    ("FD001: ظروف ثابتة (عطل واحد)", 
     "FD002: 6 ظروف متغيرة (عطل واحد)",
     "FD003: ظروف ثابتة (عطلين) - 🚧 قريباً",
     "FD004: 6 ظروف متغيرة (عطلين) - 🚧 قريباً")
)

# دالة سريعة لتجهيز أسماء العواميد
def get_column_names():
    return ['unit_nr', 'time_cycles'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

# ==========================================
# 🚀 لو اليوزر اختار FD001
# ==========================================
if condition == "FD001: ظروف ثابتة (عطل واحد)":
    st.subheader("🛠️ الموديل المستخدم: Random Forest (دقة 93%)")
    try:
        # تحميل الملفات
        model_1 = joblib.load('rf_model_fd001.pkl')
        scaler_1 = joblib.load('scaler_fd001.pkl')
        features_1 = joblib.load('features_fd001.pkl')
        test_df1 = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=get_column_names())
        
        # اليوزر يختار رقم المحرك
        unit_id = st.number_input("أدخل رقم المحرك لـ FD001 (من 1 لـ 100):", min_value=1, max_value=100, value=1)
        
        if st.button("توقع الانهيار 🔍"):
            unit_data = test_df1[test_df1['unit_nr'] == unit_id].tail(1).copy() # ناخد آخر سطر
            unit_data[features_1] = scaler_1.transform(unit_data[features_1]) # توحيد المقامات
            prediction = model_1.predict(unit_data[features_1]) # التوقع
            
            st.success(f"🚨 تحذير: المحرك رقم {unit_id} سينهار بعد حوالي: **{int(prediction[0])} دورة طيران** ✈️")
    except Exception as e:
        st.warning("⚠️ يرجى التأكد من وجود ملفات الموديل (rf_model_fd001.pkl) وملف الامتحان (test_FD001.txt) في الفولدر.")

# ==========================================
# 🚀 لو اليوزر اختار FD002 (المستوى الوحش)
# ==========================================
elif condition == "FD002: 6 ظروف متغيرة (عطل واحد)":
    st.subheader("🛠️ الموديل المستخدم: K-Means + XGBoost (دقة 84.4%)")
    try:
        # تحميل الملفات السحرية بتاعتنا
        model_2 = joblib.load('xgb_model_fd002.pkl')
        scaler_2 = joblib.load('scaler_fd002.pkl')
        kmeans_2 = joblib.load('kmeans_fd002.pkl')
        features_2 = joblib.load('features_fd002.pkl')
        test_df2 = pd.read_csv('test_FD002.txt', sep=r'\s+', header=None, names=get_column_names())
        
        # اليوزر يختار رقم المحرك
        unit_id = st.number_input("أدخل رقم المحرك لـ FD002 (من 1 لـ 259):", min_value=1, max_value=259, value=1)
        
        if st.button("توقع الانهيار 🔍"):
            # 1. نجيب داتا المحرك ده بس
            unit_data = test_df2[test_df2['unit_nr'] == unit_id].copy()
            
            # 2. نعمل Processing (المتوسط والانحراف لآخر 15 دورة)
            features_to_process = [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
            for col in features_to_process:
                unit_data[col + '_mean'] = unit_data[col].rolling(15).mean()
                unit_data[col + '_std'] = unit_data[col].rolling(15).std()
            unit_data = unit_data.bfill()
            
            # 3. ناخد سطر "الإنقاذ" الأخير
            unit_last = unit_data.tail(1).copy()
            
            # 4. الـ Scaling
            cols_to_scale = features_to_process + [col + '_mean' for col in features_to_process] + [col + '_std' for col in features_to_process]
            unit_last[cols_to_scale] = scaler_2.transform(unit_last[cols_to_scale])
            
            # 5. سحر الـ K-Means
            cluster = kmeans_2.predict(unit_last[['setting_1', 'setting_2', 'setting_3']])
            for i in range(6):
                unit_last[f'Condition_{i}'] = (cluster == i).astype(int)
                
            # 6. التوقع النهائي
            prediction = model_2.predict(unit_last[features_2])
            
            st.success(f"🚨 تحذير: المحرك رقم {unit_id} سينهار بعد حوالي: **{int(prediction[0])} دورة طيران** ✈️")
            
    except Exception as e:
         st.warning(f"⚠️ يرجى التأكد من وجود ملفات FD002. الخطأ: {e}")

# ==========================================
# 🚧 لو اليوزر اختار أي حاجة تانية
# ==========================================
else:
    st.info("🚧 جاري العمل على تدريب هذا الموديل المعقد.. انتظرونا قريباً!")