import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_data
import sys
import io

# Set page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model."""
    # Try to load advanced model first
    model_info_path = os.path.join('models', 'advanced_model_info.pkl')
    
    if os.path.exists(model_info_path):
        model_info = joblib.load(model_info_path)
        model_path = os.path.join('models', f"{model_info['model_name']}_advanced.pkl")
        model = joblib.load(model_path)
        return model, model_info, True
    
    # Fall back to regular model
    model_info_path = os.path.join('models', 'model_info.pkl')
    model_info = joblib.load(model_info_path)
    model_path = os.path.join('models', f"{model_info['model_name']}.pkl")
    model = joblib.load(model_path)
    
    return model, model_info, False

@st.cache_data
def load_dataset():
    """Load the Titanic dataset."""
    data_path = os.path.join('data', 'titanic.csv')
    return pd.read_csv(data_path)

def debug_predictions(passenger_data):
    """Debug the prediction process to identify issues."""
    # Convert single passenger to DataFrame
    passenger_df = pd.DataFrame([passenger_data])
    
    # Debug preprocessing
    debug_info = {}
    
    # Check if model info exists
    model_info_path = os.path.join('models', 'advanced_model_info.pkl')
    if os.path.exists(model_info_path):
        model_info = joblib.load(model_info_path)
        debug_info['expected_features'] = model_info.get('feature_names', [])
    
    # Load preprocessor info
    preprocessor_path = os.path.join('models', 'ensemble_preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        preprocessor_info = joblib.load(preprocessor_path)
        debug_info['categorical_features'] = preprocessor_info.get('categorical_features', [])
        debug_info['numerical_features'] = preprocessor_info.get('numerical_features', [])
    
    # Capture preprocessing steps
    try:
        # Step 1: Apply feature engineering
        from preprocess import engineer_features
        engineered_df = engineer_features(passenger_df)
        debug_info['engineered_columns'] = engineered_df.columns.tolist()
        
        # Step 2: Get one-hot encoded features
        categorical_features = debug_info.get('categorical_features', 
                                            ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'CabinDeck'])
        numerical_features = debug_info.get('numerical_features',
                                          ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 
                                           'IsAlone', 'FarePerPerson', 'HasCabin', 'TicketNumber'])
        
        dummies = pd.get_dummies(engineered_df[categorical_features], drop_first=True)
        debug_info['dummy_columns'] = dummies.columns.tolist()
        
        # Step 3: Final feature matrix
        X = pd.concat([dummies, engineered_df[numerical_features]], axis=1)
        debug_info['final_columns'] = X.columns.tolist()
        
        # Step 4: Compare with expected features
        if 'expected_features' in debug_info:
            missing_cols = set(debug_info['expected_features']) - set(debug_info['final_columns'])
            debug_info['missing_columns'] = list(missing_cols)
    
    except Exception as e:
        debug_info['error'] = str(e)
    
    return debug_info

def predict_survival(model, passenger_data):
    """Predict survival probability for a passenger."""
    # Convert single passenger to DataFrame
    passenger_df = pd.DataFrame([passenger_data])
    
    try:
        # Preprocess passenger data
        X, _ = preprocess_data(passenger_df, model_type="advanced")
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        debug_info = debug_predictions(passenger_data)
        st.expander("Debug Information").json(debug_info)
        return None, None

def capture_output(func, *args, **kwargs):
    """Capture output from a function."""
    # Redirect stdout
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    # Call function
    result = func(*args, **kwargs)
    
    # Get output
    output = new_stdout.getvalue()
    
    # Restore stdout
    sys.stdout = old_stdout
    
    return result, output

def main():
    """Main function for Streamlit app."""
    # Header
    st.title("🚢 Titanic Survival Prediction")
    st.write("Predict whether a passenger would survive the Titanic disaster.")
    
    # Load model and dataset
    model, model_info, is_advanced = load_model()
    df = load_dataset()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Make Prediction", "Dataset Overview", "Model Information"])
    
    # Model badge
    model_type = "Advanced" if is_advanced else "Basic"
    st.sidebar.success(f"Using {model_type} Model: {model_info['model_name']}")
    st.sidebar.info(f"Model Accuracy: {model_info['accuracy']:.2%}")
    
    if page == "Make Prediction":
        st.header("Passenger Information")
        
        # Create columns for form
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], 
                                  help="1 = 1st class (Upper), 2 = 2nd class (Middle), 3 = 3rd class (Lower)")
            sex = st.radio("Gender", ["male", "female"])
            age = st.slider("Age", 0.5, 80.0, 30.0)
            
        with col2:
            sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
            parch = st.slider("Number of Parents/Children Aboard", 0, 6, 0)
            fare = st.slider("Ticket Fare", 0.0, 512.0, 32.0)
            embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], 
                                   help="C = Cherbourg, Q = Queenstown, S = Southampton")
        
        # Additional fields for completeness
        title = "Mr" if sex == "male" else "Mrs" if age >= 18 else "Miss"
        name = f"Passenger, {title}. Test"
        
        # Create passenger data
        passenger_data = {
            'PassengerId': 1000,
            'Pclass': pclass,
            'Name': name,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Ticket': 'TEST1234',
            'Fare': fare,
            'Cabin': '',
            'Embarked': embarked
        }
        
        # Make prediction
        st.subheader("Prediction")
        if st.button("Predict Survival"):
            prediction, probability = predict_survival(model, passenger_data)
            
            # Check if prediction was successful
            if prediction is not None and probability is not None:
                # Create columns for result
                result_col, viz_col = st.columns([1, 1])
                
                with result_col:
                    if prediction == 1:
                        st.success(f"✅ This passenger would likely **SURVIVE** the Titanic disaster.")
                    else:
                        st.error(f"❌ This passenger would likely **NOT SURVIVE** the Titanic disaster.")
                        
                    st.write(f"Survival Probability: **{probability:.2%}**")
                    
                    # Factors influencing survival
                    st.subheader("Key factors for this prediction:")
                    factors = []
                    
                    if sex == "female":
                        factors.append("👉 Women had higher survival rates (74% vs 19% for men)")
                    else:
                        factors.append("👉 Men had lower survival rates (19% vs 74% for women)")
                        
                    if pclass == 1:
                        factors.append("👉 First class passengers had higher survival rates (63%)")
                    elif pclass == 3:
                        factors.append("👉 Third class passengers had lower survival rates (24%)")
                        
                    if age < 10:
                        factors.append("👉 Children under 10 had higher survival rates")
                        
                    family_size = sibsp + parch + 1
                    if family_size > 4:
                        factors.append("👉 Very large families had lower survival rates")
                    elif family_size > 1:
                        factors.append("👉 Small families had higher survival rates than solo travelers")
                    
                    for factor in factors:
                        st.write(factor)
                
                with viz_col:
                    # Create gauge chart for probability
                    fig, ax = plt.subplots(figsize=(4, 0.3))
                    ax.barh([0], [probability], color='green', height=0.3)
                    ax.barh([0], [1-probability], left=[probability], color='red', height=0.3)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_yticks([])
                    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
                    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
                    st.pyplot(fig)
                    
                    # Create passenger profile
                    st.write("**Passenger Profile:**")
                    profile = f"""
                    - **Class:** {pclass}
                    - **Gender:** {sex.capitalize()}
                    - **Age:** {age} years
                    - **Family:** {sibsp} siblings/spouse, {parch} parents/children
                    - **Fare:** ${fare:.2f}
                    - **Embarked:** {embarked}
                    """
                    st.write(profile)
            else:
                st.warning("Could not make a prediction. Please check the debug information above.")
        
    elif page == "Dataset Overview":
        st.header("Titanic Dataset Overview")
        
        # Show dataset stats
        st.subheader("Dataset Statistics")
        st.write(f"Number of passengers: {len(df)}")
        st.write(f"Survival rate: {df['Survived'].mean():.2%}")
        
        # Show first few rows
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Show basic stats
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
        
        # Visualizations
        st.subheader("Visualizations")
        viz_type = st.selectbox("Select Visualization", [
            "Survival Distribution", 
            "Survival by Gender", 
            "Survival by Class",
            "Age Distribution",
            "Fare Distribution",
            "Correlation Matrix"
        ])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "Survival Distribution":
            sns.countplot(x='Survived', hue='Survived', data=df, palette=['#FF9999', '#66B2FF'], ax=ax, legend=False)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Did Not Survive', 'Survived'])
            ax.set_title('Survival Distribution')
            
        elif viz_type == "Survival by Gender":
            sns.countplot(x='Sex', hue='Survived', data=df, palette=['#FF9999', '#66B2FF'], ax=ax)
            ax.set_xticklabels(['Male', 'Female'])
            ax.set_title('Survival by Gender')
            ax.legend(['Did Not Survive', 'Survived'])
            
        elif viz_type == "Survival by Class":
            sns.countplot(x='Pclass', hue='Survived', data=df, palette=['#FF9999', '#66B2FF'], ax=ax)
            ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
            ax.set_title('Survival by Passenger Class')
            ax.legend(['Did Not Survive', 'Survived'])
            
        elif viz_type == "Age Distribution":
            sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=20, 
                       palette=['#FF9999', '#66B2FF'], ax=ax)
            ax.set_title('Age Distribution by Survival')
            ax.legend(['Did Not Survive', 'Survived'])
            
        elif viz_type == "Fare Distribution":
            sns.histplot(data=df, x='Fare', hue='Survived', multiple='stack', bins=20, 
                       palette=['#FF9999', '#66B2FF'], ax=ax)
            ax.set_title('Fare Distribution by Survival')
            ax.legend(['Did Not Survive', 'Survived'])
            
        elif viz_type == "Correlation Matrix":
            numeric_df = df.select_dtypes(include=[np.number])
            correlation = numeric_df.corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Correlation Matrix')
        
        st.pyplot(fig)
        
    else:  # Model Information
        st.header("Model Information")
        
        # Show model type
        st.subheader("Model Details")
        st.write(f"Model type: **{model_info['model_name']}**")
        st.write(f"Accuracy: **{model_info['accuracy']:.2%}**")
        
        # Feature importance (if available)
        if os.path.exists(os.path.join('models', 'enhanced_feature_importance.png')):
            st.subheader("Enhanced Feature Importance Analysis")
            st.image(os.path.join('models', 'enhanced_feature_importance.png'))
            st.write("""
            This visualization shows four different perspectives on feature importance:
            1. Random Forest Feature Importances: Shows how much each feature contributes to the model's predictions
            2. Permutation Importances: Measures how model performance decreases when a feature is randomly shuffled
            3. Feature Correlation with Target: Shows how strongly each feature correlates with survival
            4. Top 5 Features Distribution: Shows how the most important features are distributed between survivors and non-survivors
            """)
        elif os.path.exists(os.path.join('models', 'rf_feature_importance.png')):
            st.subheader("Feature Importance")
            st.image(os.path.join('models', 'rf_feature_importance.png'))
        
        # Add model accuracy comparison if available
        if os.path.exists(os.path.join('models', 'model_accuracy_comparison.png')):
            st.subheader("Model Accuracy Comparison")
            st.image(os.path.join('models', 'model_accuracy_comparison.png'))
            st.write("""
            This chart compares the accuracy of different machine learning models:
            - Random Forest: Uses multiple decision trees to make predictions
            - Gradient Boosting: Builds trees sequentially to correct errors from previous trees
            - SVC (Support Vector Classifier): Finds the optimal boundary between classes
            - Logistic Regression: Uses probability to predict binary outcomes
            - Ensemble: Combines all the above models to make even better predictions
            """)
        
        # Add ROC curve comparison if available
        if os.path.exists(os.path.join('models', 'roc_curve_comparison.png')):
            st.subheader("ROC Curve Comparison")
            st.image(os.path.join('models', 'roc_curve_comparison.png'))
            st.write("""
            The ROC (Receiver Operating Characteristic) curve shows the tradeoff between true positive rate and 
            false positive rate at different classification thresholds. A higher Area Under the Curve (AUC) 
            indicates better model performance.
            """)
        
        # Model performance
        if os.path.exists(os.path.join('models', f'confusion_matrix_{model_info["model_name"]}.png')):
            st.subheader("Model Performance")
            
            # Create columns for metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Confusion Matrix**")
                st.image(os.path.join('models', f'confusion_matrix_{model_info["model_name"]}.png'))
            
            with col2:
                st.write("**ROC Curve**")
                if os.path.exists(os.path.join('models', f'roc_curve_{model_info["model_name"]}.png')):
                    st.image(os.path.join('models', f'roc_curve_{model_info["model_name"]}.png'))
                else:
                    st.write("ROC curve not available for this model.")
        
        # Technical information
        with st.expander("Technical Model Parameters"):
            st.json(model_info['parameters'])

if __name__ == "__main__":
    main() 