import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Dự đoán Điểm số Học sinh (G3)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thiết kế giao diện Header
st.write("""
# 🎓 Hệ Thống Phân Tích & Dự Đoán Kết Quả Học Tập G3
Ứng dụng Machine Learning hỗ trợ dự đoán điểm cuối kỳ (G3) của học sinh dựa trên các thông tin cá nhân, gia đình và điểm quá trình (G1, G2).
""")

# Khởi tạo thanh Sidebar điều hướng
st.sidebar.title("⚙️ Bảng Điều Khiển")
app_mode = st.sidebar.selectbox("Chọn trang làm việc", ["Tải dữ liệu & Huấn luyện", "Trực quan hóa & Đánh giá", "Giao diện Dự đoán Tương tác"])

# Hàm tải dữ liệu và cache
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Lỗi định dạng file: {e}")
        return None

# Hàm huấn luyện mô hình và cache tài nguyên
@st.cache_resource
def train_model(df_train, use_g1_g2=True):
    target_col = 'G3'
    if use_g1_g2:
        feature_cols = [col for col in df_train.columns if col != target_col]
    else:
        feature_cols = [col for col in df_train.columns if col not in [target_col, 'G1', 'G2']]
        
    X_train = df_train[feature_cols].copy()
    y_train = df_train[target_col].copy()
    
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model_pipeline.fit(X_train, y_train)
    return model_pipeline, feature_cols, num_cols, cat_cols

# Phần tải dữ liệu chung
st.sidebar.header("📁 Nhập Dữ Liệu")
train_file = st.sidebar.file_uploader("Tải file Train_Dataset.xlsx (bắt buộc)", type=["xlsx"])
test_file = st.sidebar.file_uploader("Tải file Test_Dataset.xlsx (Tùy chọn)", type=["xlsx"])

if train_file is not None:
    df_train = load_data(train_file)
    st.session_state['df_train'] = df_train
    
    # Tự động huấn luyện mô hình
    model_pipeline, feature_cols, num_cols, cat_cols = train_model(df_train)
    st.session_state['model_pipeline'] = model_pipeline
    st.session_state['feature_cols'] = feature_cols
    st.session_state['num_cols'] = num_cols
    st.session_state['cat_cols'] = cat_cols
    
    if test_file is not None:
        df_test = load_data(test_file)
        st.session_state['df_test'] = df_test
else:
    st.warning("⚠️ Vui lòng tải file dữ liệu huấn luyện (Train_Dataset.xlsx) tại thanh Sidebar để kích hoạt hệ thống.")

# --- TRANG 1: TẢI DỮ LIỆU & HUẤN LUYỆN ---
if app_mode == "Tải dữ liệu & Huấn luyện":
    st.subheader("📊 Quản Lý & Xem Trước Dữ Liệu")
    if 'df_train' in st.session_state:
        df_train = st.session_state['df_train']
        st.success("✅ Tải dữ liệu Train thành công!")
        st.write(f"**Kích thước tập huấn luyện:** {df_train.shape[0]} dòng, {df_train.shape[1]} cột.")
        st.dataframe(df_train.head(10))
        
        # Kiểm tra trùng lặp và thiếu giá trị
        dup_count = df_train.duplicated().sum()
        missing_count = df_train.isnull().sum().sum()
        col1, col2 = st.columns(2)
        col1.metric("Số dòng trùng lặp", dup_count)
        col2.metric("Tổng số ô bị trống (Null)", missing_count)
    else:
        st.info("Vui lòng tải lên file dữ liệu ở thanh bên để tiếp tục.")

# --- TRANG 2: TRỰC QUAN HÓA & ĐÁNH GIÁ ---
elif app_mode == "Trực quan hóa & Đánh giá":
    st.subheader("📈 Đánh Giá Hiệu Năng Mô Hình Hồi Quy")
    if 'model_pipeline' in st.session_state and 'df_train' in st.session_state:
        model_pipeline = st.session_state['model_pipeline']
        df_train = st.session_state['df_train']
        feature_cols = st.session_state['feature_cols']
        num_cols = st.session_state['num_cols']
        cat_cols = st.session_state['cat_cols']
        
        # Đánh giá bằng tập Test
        if 'df_test' in st.session_state:
            df_test = st.session_state['df_test']
            X_test = df_test[feature_cols]
            y_test = df_test['G3']
            y_pred = model_pipeline.predict(X_test)
            
            # Tính chỉ số lỗi
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)
            
            st.markdown("### 📊 Chỉ số kiểm nghiệm trên tập Test độc lập")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{mae:.4f}")
            c2.metric("MSE", f"{mse:.4f}")
            c3.metric("RMSE", f"{rmse:.4f}")
            c4.metric("R² Score", f"{r2:.4f}")
            
            # Biểu đồ đánh giá
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            # 1. Thực tế vs Dự đoán
            sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], alpha=0.7, color='teal')
            axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0].set_title('Thực tế vs Dự đoán G3')
            # 2. Phân phối Sai số
            residuals = y_test - y_pred
            sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
            axes[1].axvline(x=0, color='red', linestyle='--')
            axes[1].set_title('Biểu đồ phân phối sai số (Residuals)')
            st.pyplot(fig)
        else:
            st.info("💡 Gợi ý: Hãy tải lên cả file Test_Dataset.xlsx ở thanh bên để hiển thị biểu đồ phân tích sai số chi tiết.")

        # Phân tích mức độ quan trọng hệ số Beta
        st.markdown("### 🔑 Hệ số Hồi Quy & Các thuộc tính tác động nhiều nhất")
        regressor_model = model_pipeline.named_steps['regressor']
        cat_encoder = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols).tolist()
        all_features = num_cols + encoded_cat_cols
        coefficients = regressor_model.coef_
        
        coef_df = pd.DataFrame({
            'Thuộc tính': all_features,
            'Hệ số Beta': coefficients,
            'Trị tuyệt đối': np.abs(coefficients)
        }).sort_values(by='Trị tuyệt đối', ascending=False).reset_index(drop=True)
        
        fig_coef, ax_coef = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Hệ số Beta', y='Thuộc tính', data=coef_df.head(10), ax=ax_coef, palette='coolwarm')
        ax_coef.set_title('Top 10 thuộc tính ảnh hưởng lớn nhất đến điểm G3')
        st.pyplot(fig_coef)
    else:
        st.info("Vui lòng cung cấp tệp dữ liệu huấn luyện để bắt đầu phân tích.")

# --- TRANG 3: GIAO DIỆN DỰ ĐOÁN TƯƠNG TÁC ---
elif app_mode == "Giao diện Dự đoán Tương tác":
    st.subheader("🔮 Công Cụ Dự Đoán Điểm Số G3")
    if 'model_pipeline' in st.session_state:
        model_pipeline = st.session_state['model_pipeline']
        feature_cols = st.session_state['feature_cols']
        df_train = st.session_state['df_train']
        
        st.write("Điền thông tin học sinh dưới đây để mô hình dự báo điểm số kết quả:")
        
        input_data = {}
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### 📝 Kết quả Học tập & Sức khỏe")
            input_data['G1'] = st.slider("Điểm thi Giữa kỳ 1 (G1)", 0, 20, int(df_train['G1'].median()))
            input_data['G2'] = st.slider("Điểm thi Giữa kỳ 2 (G2)", 0, 20, int(df_train['G2'].median()))
            input_data['studytime'] = st.slider("Thời gian học tập (1: <2h, 4: >10h)", 1, 4, int(df_train['studytime'].median()))
            input_data['failures'] = st.slider("Số lần thi trượt trong quá khứ", 0, 3, int(df_train['failures'].median()))
            input_data['absences'] = st.slider("Số buổi nghỉ học", 0, 93, int(df_train['absences'].median()))
            
        with col_right:
            st.markdown("#### 🏫 Thông tin Cá nhân & Gia đình")
            input_data['school'] = st.selectbox("Trường học", df_train['school'].unique())
            input_data['sex'] = st.selectbox("Giới tính", df_train['sex'].unique())
            input_data['age'] = st.number_input("Tuổi tác", 15, 22, int(df_train['age'].median()))
            input_data['address'] = st.selectbox("Khu vực sinh sống (U: Đô thị, R: Nông thôn)", df_train['address'].unique())
            input_data['internet'] = st.selectbox("Truy cập Internet tại nhà", df_train['internet'].unique())
            
        # Gán giá trị mặc định cho các cột còn lại để tránh lỗi mô hình
        for col in feature_cols:
            if col not in input_data:
                if df_train[col].dtype in ['int64', 'float64']:
                    input_data[col] = df_train[col].median()
                else:
                    input_data[col] = df_train[col].dropna().unique()[0]
                    
        if st.button("🎯 Dự Đoán Kết Quả", type="primary"):
            input_df = pd.DataFrame([input_data])
            prediction = model_pipeline.predict(input_df)[0]
            final_score = max(0.0, min(20.0, prediction))
            
            st.markdown(f"""
            <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 6px solid #1e90ff;'>
                <h3 style='margin: 0; color: #1e90ff;'>🎯 Điểm G3 Dự Kiến:</h3>
                <p style='font-size: 36px; font-weight: bold; margin: 10px 0 0 0; color: #104e8b;'>{final_score:.2f} / 20</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Vui lòng tải lên tệp Train_Dataset.xlsx để kích hoạt tính năng dự đoán.")
