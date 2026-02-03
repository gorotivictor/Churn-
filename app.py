import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import xgboost as xgb  
import pickle
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Color palette
COLORS = {
    'primary': '#14213d',      # Dark blue
    'secondary': '#fca311',    # Orange
    'background': '#ffffff',   # White
    'light_gray': '#e5e5e5',  # Light gray
    'black': '#000000'         # Black
}

# Custom CSS
def apply_custom_css():
    st.markdown(f"""
    <style>
    .main {{
        background-color: {COLORS['background']};
    }}
    
    /* Force all text to be black */
    .stApp, .main, .block-container {{
        color: {COLORS['black']} !important;
    }}
    
    /* Override Streamlit's default text colors */
    h1, h2, h3, h4, h5, h6 {{
        color: {COLORS['light_gray']} !important;
    }}
    
  
    p, div, span {{
        color: {COLORS['black']} !important;
    }}
    
    /* Input fields and labels */
    .stTextInput > label, .stSelectbox > label, .stNumberInput > label {{
        color: {COLORS['black']} !important;
        font-weight: bold;
    }}
    
    .stTextInput input, .stSelectbox select, .stNumberInput input {{
        color: {COLORS['light_gray']} !important;
    }}
    
    /* Success/Error messages */
    .stSuccess, .stError, .stWarning, .stInfo {{
        color: {COLORS['black']} !important;
    }}
    
    .stSuccess div, .stError div, .stWarning div, .stInfo div {{
        color: {COLORS['black']} !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['secondary']};
        color: {COLORS['primary']};
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }}
    
    .stButton > button:hover {{
        background-color: {COLORS['primary']};
        color: {COLORS['secondary']};
    }}
    
    /* Metric cards */
    .metric-card {{
        background-color: {COLORS['light_gray']};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {COLORS['secondary']};
        margin: 10px 0;
        color: {COLORS['black']} !important;
    }}
    
    .metric-card h2, .metric-card h3 {{
        color: {COLORS['primary']} !important;
    }}
    
    /* Prediction results */
    .prediction-result {{
        background-color: {COLORS['primary']};
        color: {COLORS['background']} !important;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }}
    
    .prediction-result h2, .prediction-result h3 {{
        color: {COLORS['background']} !important;
    }}
    
    /* Header text */
    .header-text {{
        color: {COLORS['primary']} !important;
        font-weight: bold;
    }}
    
    /* Sidebar text */
    .css-1d391kg, .css-1lcbmhc {{
        color: {COLORS['light_gray']} !important;
    }}
    
    /* Dataframe text */
    .dataframe {{
        color: {COLORS['black']} !important;
    }}
    
    /* Tab labels */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
        color: {COLORS['light_gray']} !important;
    }}
    
    /* Markdown text */
    .stMarkdown {{
        color: {COLORS['light_gray']} !important;
    }}
    
    /* File uploader */
    .stFileUploader > label {{
        color: {COLORS['black']} !important;
    }}
    
    /* Multiselect */
    .stMultiSelect > label {{
        color: {COLORS['black']} !important;
    }}
    
    /* Slider */
    .stSlider > label {{
        color: {COLORS['light_gray']} !important;
    }}
    
    /* Checkbox */
    .stCheckbox > label {{
        color: {COLORS['black']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None

# Login page
def login_page():
    st.markdown('<h1 class="header-text">🏦 Sunrise Microfinance Bank</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="header-text">Customer Churn Prediction System</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Admin Login")
        username = st.text_input("Username", placeholder="Enter admin username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        if st.button("Login", use_container_width=True):
            # Simple authentication (in production, use proper authentication)
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.success("Login successful!")
            else:
                st.error("Invalid credentials. Use admin/admin123")

# Simple oversampling function to replace SMOTE
def simple_oversample(X, y, random_state=42):
    """Simple oversampling by duplicating minority class samples"""
    np.random.seed(random_state)
    
    # Combine features and target
    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    
    # Separate majority and minority classes
    majority_class = df[df[y.name] == 0]
    minority_class = df[df[y.name] == 1]
    
    # Oversample minority class
    minority_upsampled = resample(minority_class, 
                                 replace=True,
                                 n_samples=len(majority_class),
                                 random_state=random_state)
    
    # Combine majority and upsampled minority
    df_upsampled = pd.concat([majority_class, minority_upsampled])
    
    # Separate features and target
    X_resampled = df_upsampled.drop(y.name, axis=1)
    y_resampled = df_upsampled[y.name]
    
    return X_resampled, y_resampled

# Data preprocessing function
def preprocess_data(df):
    # Drop non-predictive columns
    if 'CustomerId' in df.columns:
        df = df.drop(['CustomerId'], axis=1)
    if 'Surname' in df.columns:
        df = df.drop(['Surname'], axis=1)
    
    # Feature encoding
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Account Activity'] = df['Account Activity'].map({'Active': 0, 'Dormant': 1})
    df['Repayment Timeliness'] = df['Repayment Timeliness'].map({'On-time': 0, 'Late': 1})
    
    df['Account Balance Trend'] = df['Account Balance Trend'].map({
        'Decreasing': 0,
        'Stable': 1,
        'Increasing': 2
    })
    
    # Convert binary columns to int
    binary_columns = ['Use of Savings Products', 'Use of Loan Products', 'Participation in Group Lending']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # One-hot encoding for categorical variables
    categorical_columns = ['Marital Status', 'Education Level', 'Loan History', 'Use of Digital Banking']
    for col in categorical_columns:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], prefix=col.replace(' ', '_'))
    
    return df

# Dashboard page
def dashboard_page():
    st.markdown('<h1 class="header-text">📊 Super Admin Dashboard</h1>', unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Customers</h3>
                <h2>{len(df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            churn_rate = df['Exited'].mean() * 100 if 'Exited' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Churn Rate</h3>
                <h2>{churn_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            active_customers = len(df) - df['Exited'].sum() if 'Exited' in df.columns else len(df)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Active Customers</h3>
                <h2>{active_customers}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_age = df['Age'].mean() if 'Age' in df.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Average Age</h3>
                <h2>{avg_age:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        st.markdown("### 📈 Customer Analytics")
        
        if 'Exited' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Churn distribution
                churn_counts = df['Exited'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=['Retained', 'Churned'],
                    values=[churn_counts[0], churn_counts[1]],
                    marker_colors=[COLORS['secondary'], COLORS['primary']]
                )])
                fig.update_layout(title="Customer Retention vs Churn", title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age distribution by churn
                fig = px.histogram(df, x='Age', color='Exited', nbins=20,
                                 title="Age Distribution by Churn Status",
                                 color_discrete_map={0: COLORS['secondary'], 1: COLORS['primary']})
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please upload data first to see dashboard metrics.")

# Upload data page
def upload_data_page():
    st.markdown('<h1 class="header-text">📁 Upload Customer Data</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your customer dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Data uploaded successfully! {len(df)} records loaded.")
            
            # Display data info
            st.markdown("### Data Preview")
            st.dataframe(df.head(10))
            
            st.markdown("### Data Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset Shape:**")
                st.write(f"Rows: {df.shape[0]}")
                st.write(f"Columns: {df.shape[1]}")
            
            with col2:
                st.markdown("**Missing Values:**")
                missing_values = df.isnull().sum().sum()
                st.write(f"Total: {missing_values}")
            
            # Store data in session state
            st.session_state.data = df
            
            if st.button("Process Data", use_container_width=True):
                with st.spinner("Processing data..."):
                    processed_df = preprocess_data(df.copy())
                    st.session_state.processed_data = processed_df
                    st.success("Data processed successfully!")
                    st.markdown("### Processed Data Preview")
                    st.dataframe(processed_df.head())
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    # Sample data option
    st.markdown("### Or Use Sample Data")
    if st.button("Load Sample Data"):
        # Create sample data based on your description
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'CustomerId': [f'SMB{15565700 + i + 1}' for i in range(n_samples)],
            'Surname': ['Abdullahi', 'Bello', 'Adesina', 'Sule', 'Nwachukwu'] * (n_samples // 5),
            'Age': np.random.randint(18, 92, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Marital Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'Education Level': np.random.choice(['None', 'Primary', 'Secondary', 'Tertiary'], n_samples),
            'Account Balance Trend': np.random.choice(['Decreasing', 'Stable', 'Increasing'], n_samples),
            'Loan History': np.random.choice(['Active', 'Cleared', 'Defaulted'], n_samples),
            'Frequency of Deposits/Withdrawals': np.random.randint(0, 15, n_samples),
            'Average Transaction Value': np.random.uniform(1000, 50000, n_samples),
            'Account Activity': np.random.choice(['Active', 'Dormant'], n_samples),
            'Use of Savings Products': np.random.choice([0, 1], n_samples),
            'Use of Loan Products': np.random.choice([0, 1], n_samples),
            'Use of Digital Banking': np.random.choice(['USSD', 'App', 'Both', 'None'], n_samples),
            'Participation in Group Lending': np.random.choice([0, 1], n_samples),
            'Tenure': np.random.randint(0, 10, n_samples),
            'Number of Complaints Logged': np.random.randint(0, 5, n_samples),
            'Response Time to Complaints': np.random.randint(0, 15, n_samples),
            'Customer Support Interactions': np.random.randint(0, 10, n_samples),
            'Repayment Timeliness': np.random.choice(['On-time', 'Late'], n_samples),
            'Overdue Loan Frequency': np.random.randint(0, 5, n_samples),
            'Penalties Paid': np.random.uniform(0, 10000, n_samples),
            'Exited': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(sample_data)
        st.session_state.data = df
        st.success("Sample data loaded successfully!")
        st.dataframe(df.head())

# Model training page
def model_training_page():
    st.markdown('<h1 class="header-text">🤖 Model Training</h1>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("Please upload data first.")
        return
    
    df = st.session_state.data.copy()
    
    st.markdown("### Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.3, 0.05)
        use_oversampling = st.checkbox("Use Oversampling for Imbalanced Data", value=True)

    with col2:
        random_state = st.number_input("Random State", value=42)
    
    selected_features = st.multiselect(
        "Select Features for Training",
        ['Age', 'Gender', 'Tenure', 'Frequency of Deposits/Withdrawals', 
         'Repayment Timeliness', 'Account Activity', 'Account Balance Trend'],
        default=['Age', 'Gender', 'Tenure', 'Frequency of Deposits/Withdrawals', 
                'Repayment Timeliness', 'Account Activity', 'Account Balance Trend']
    )
    
    if st.button("Train Models", use_container_width=True):
        if not selected_features:
            st.error("Please select at least one feature.")
            return
            
        with st.spinner("Training models..."):
            # Preprocess data
            processed_df = preprocess_data(df)
            
            # Prepare features and target
            available_features = [f for f in selected_features if f in processed_df.columns]
            X = processed_df[available_features]
            y = processed_df['Exited']
            
            # Handle class imbalance with SMOTE
            if use_oversampling:
                X_resampled, y_resampled = simple_oversample(X, y, random_state=random_state)
            else:
                X_resampled, y_resampled = X, y
            # Feature scaling
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_resampled)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_resampled, test_size=test_size, random_state=random_state
            )
            
            # Train models
            models = {
                'Logistic Regression': LogisticRegression(random_state=random_state),
                'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100),
                'XGBoost': xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
            }
            
            results = {}
            trained_models = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                results[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'F1-Score': f1_score(y_test, y_pred),
                    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
                }
                trained_models[name] = model
            
            # Select best model
            best_model_name = max(results, key=lambda x: results[x]['F1-Score'])
            best_model = trained_models[best_model_name]
            
            # Store in session state
            st.session_state.model = best_model
            st.session_state.scaler = scaler
            st.session_state.model_results = results
            st.session_state.best_model_name = best_model_name
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.model_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            st.success(f"Models trained successfully! Best model: {best_model_name}")
            
            # Display results
            st.markdown("### Model Performance")
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df.round(4))
            
            # Feature importance
            if best_model_name in ['Random Forest', 'XGBoost']:
                st.markdown("### Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title="Feature Importance",
                           color_discrete_sequence=[COLORS['secondary']])
                st.plotly_chart(fig, use_container_width=True)

# Prediction page
def prediction_page():
    st.markdown('<h1 class="header-text">🔮 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first.")
        return
    
    tab1, tab2 = st.tabs(["Single Prediction", "Bulk Prediction"])
    
    with tab1:
        st.markdown("### Single Customer Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 18, 100, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            tenure = st.number_input("Tenure (years)", 0, 10, 2)
            freq_deposits = st.number_input("Frequency of Deposits/Withdrawals", 0, 14, 5)
        
        with col2:
            repayment = st.selectbox("Repayment Timeliness", ["On-time", "Late"])
            account_activity = st.selectbox("Account Activity", ["Active", "Dormant"])
            balance_trend = st.selectbox("Account Balance Trend", ["Decreasing", "Stable", "Increasing"])
        
        if st.button("Predict Churn", use_container_width=True):
            # Prepare input data
            input_data = pd.DataFrame({
                'Age': [age / 100],  # Normalized
                'Gender': [1 if gender == "Female" else 0],
                'Tenure': [tenure / 10],  # Normalized
                'Frequency of Deposits/Withdrawals': [freq_deposits / 14],  # Normalized
                'Repayment Timeliness': [1 if repayment == "Late" else 0],
                'Account Activity': [1 if account_activity == "Dormant" else 0],
                'Account Balance Trend': [0 if balance_trend == "Decreasing" else 1 if balance_trend == "Stable" else 2]
            })
            
            # Make prediction
            prediction = st.session_state.model.predict(input_data)[0]
            probability = st.session_state.model.predict_proba(input_data)[0]
            
            # Display result
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-result" style="background-color: {COLORS['primary']};">
                    <h2>⚠️ HIGH CHURN RISK</h2>
                    <h3>Probability: {probability[1]:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-result" style="background-color: {COLORS['secondary']};">
                    <h2>✅ LOW CHURN RISK</h2>
                    <h3>Probability: {probability[0]:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Key factors
            st.markdown("### Key Risk Factors")
            risk_factors = []
            if age < 30 or age > 70:
                risk_factors.append("Age group has higher churn tendency")
            if account_activity == "Dormant":
                risk_factors.append("Dormant account increases churn risk")
            if repayment == "Late":
                risk_factors.append("Late repayments indicate financial stress")
            if freq_deposits < 3:
                risk_factors.append("Low transaction frequency")
            if tenure < 2:
                risk_factors.append("Short tenure with bank")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("• Customer profile shows good retention indicators")
    
    with tab2:
        st.markdown("### Bulk Prediction")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for bulk prediction",
            type=['csv'],
            help="Upload a CSV file with customer data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} records")
                
                if st.button("Run Bulk Prediction"):
                    # Process and predict
                    processed_df = preprocess_data(df.copy())
                    
                    # Ensure all required features are present
                    required_features = st.session_state.feature_names
                    available_features = [f for f in required_features if f in processed_df.columns]
                    
                    if len(available_features) == len(required_features):
                        X = processed_df[available_features]
                        X_scaled = st.session_state.scaler.transform(X)
                        
                        predictions = st.session_state.model.predict(X_scaled)
                        probabilities = st.session_state.model.predict_proba(X_scaled)[:, 1]
                        
                        # Add results to dataframe
                        results_df = df.copy()
                        results_df['Churn_Prediction'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                        results_df['Churn_Probability'] = probabilities
                        
                        st.markdown("### Prediction Results")
                        st.dataframe(results_df)
                        
                        # Summary
                        high_risk_count = sum(predictions)
                        st.markdown(f"**Summary:** {high_risk_count} out of {len(df)} customers are at high risk of churn ({high_risk_count/len(df)*100:.1f}%)")
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "churn_predictions.csv",
                            "text/csv"
                        )
                    else:
                        st.error("Missing required features in uploaded data")
                        
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Reports page
def reports_page():
    st.markdown('<h1 class="header-text">📊 Model Reports</h1>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first to view reports.")
        return
    
    # Model performance summary
    st.markdown("### Model Performance Summary")
    results_df = pd.DataFrame(st.session_state.model_results).T
    st.dataframe(results_df.round(4))
    
    # Best model info
    st.info(f"Best Model: {st.session_state.best_model_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature importance
        if st.session_state.best_model_name in ['Random Forest', 'XGBoost']:
            st.markdown("### Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                       orientation='h',
                       color_discrete_sequence=[COLORS['secondary']])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion matrix
        st.markdown("### Confusion Matrix")
        if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            cm = confusion_matrix(st.session_state.y_test, y_pred)
            
            fig = px.imshow(cm, 
                          text_auto=True, 
                          aspect="auto",
                          color_continuous_scale='Blues',
                          labels=dict(x="Predicted", y="Actual"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("### Business Recommendations")
    recommendations = [
        "Focus retention efforts on customers with short tenure and low transaction frequency",
        "Implement proactive engagement for dormant accounts",
        "Develop targeted programs for high-risk age groups",
        "Improve digital banking adoption to increase engagement",
        "Monitor and address late payment patterns early",
        "Create loyalty programs for long-term customers"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

# Main app
def main():
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    init_session_state()
    
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    pages = {
        "🏠 Dashboard": dashboard_page,
        "📁 Upload Data": upload_data_page,
        "🤖 Train Model": model_training_page,
        "🔮 Predictions": prediction_page,
        "📊 Reports": reports_page
    }
    
    selected_page = st.sidebar.selectbox("Choose a page", list(pages.keys()))
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
    
    # Display selected page
    pages[selected_page]()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sunrise Microfinance Bank**")
    st.sidebar.markdown("Customer Churn Prediction System")

if __name__ == "__main__":
    main()