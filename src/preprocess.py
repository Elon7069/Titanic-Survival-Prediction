import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os
import re

def load_data(filepath):
    """Load the Titanic dataset."""
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def extract_title(name):
    """Extract title from name."""
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def engineer_features(df):
    """Engineer new features from existing data."""
    # Create a copy of the dataframe
    df = df.copy()
    
    # Extract titles from names
    df['Title'] = df['Name'].apply(extract_title)
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create is_alone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Create fare per person feature
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 12, 18, 35, 50, 80],
                           labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Elder'])
    
    # Create fare groups
    if len(df['Fare'].unique()) >= 4:
        try:
            df['FareGroup'] = pd.qcut(df['Fare'], 
                                  q=4,
                                  labels=['Low', 'Medium', 'High', 'Very High'],
                                  duplicates='drop')
        except ValueError:
            # For single passengers or when all fares are identical
            fare = df['Fare'].iloc[0]
            if fare <= 7.91:
                df['FareGroup'] = 'Low'
            elif fare <= 14.454:
                df['FareGroup'] = 'Medium'
            elif fare <= 31.0:
                df['FareGroup'] = 'High'
            else:
                df['FareGroup'] = 'Very High'
    else:
        # For single passengers or when all fares are identical
        fare = df['Fare'].iloc[0]
        if fare <= 7.91:
            df['FareGroup'] = 'Low'
        elif fare <= 14.454:
            df['FareGroup'] = 'Medium'
        elif fare <= 31.0:
            df['FareGroup'] = 'High'
        else:
            df['FareGroup'] = 'Very High'
    
    # Create cabin features
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['CabinDeck'] = df['Cabin'].str[0].fillna('U')
    
    # Create ticket features
    df['TicketPrefix'] = df['Ticket'].str.extract('([A-Za-z]+)').fillna('')
    df['TicketNumber'] = df['Ticket'].str.extract('(\\d+)').fillna('0').astype(int)
    
    return df

def preprocess_data(df, model_type="advanced"):
    """Preprocess the data for model training or prediction."""
    print("Preprocessing data...")
    
    # Handle missing values (avoiding inplace warnings)
    df = df.copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # Engineer features
    df = engineer_features(df)
    
    # Check if we're using the basic or advanced model
    if model_type == "basic":
        # For basic model, use simpler feature set
        categorical_features = ['Sex', 'Embarked']
        numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    else:
        # For advanced model, use all engineered features
        categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'CabinDeck']
        numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 
                             'IsAlone', 'FarePerPerson', 'HasCabin', 'TicketNumber']
    
    # If the preprocessing is for the ensemble model, check if we can load the preprocessor
    if model_type == "advanced" and os.path.exists(os.path.join('models', 'ensemble_preprocessor.pkl')):
        # Load the ensemble preprocessor
        preprocessor_info = joblib.load(os.path.join('models', 'ensemble_preprocessor.pkl'))
        
        # Create feature matrix using the same features as in training
        # Use get_dummies with dummy_na=True to ensure all expected columns are created
        # This ensures that even if a category isn't present in new data, it will get a column
        dummies = pd.get_dummies(df[categorical_features], drop_first=True)
        
        # Load model info to get expected feature names
        if os.path.exists(os.path.join('models', 'advanced_model_info.pkl')):
            model_info = joblib.load(os.path.join('models', 'advanced_model_info.pkl'))
            expected_features = model_info.get('feature_names', [])
            
            # Add missing columns with zeros
            for feature in expected_features:
                if feature not in dummies.columns and feature not in numerical_features:
                    dummies[feature] = 0
        
        # Ensure proper column order
        X = pd.concat([dummies, df[numerical_features]], axis=1)
        
        # Make sure we have all necessary columns in the right order
        if os.path.exists(os.path.join('models', 'advanced_model_info.pkl')):
            model_info = joblib.load(os.path.join('models', 'advanced_model_info.pkl'))
            expected_features = model_info.get('feature_names', [])
            
            # Reindex to ensure right order and all columns
            if expected_features:
                missing_cols = set(expected_features) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X = X[expected_features]
        
        # Scale features
        scaler = preprocessor_info['scaler']
        X_scaled = scaler.transform(X)
        
        return X_scaled, None
    
    # Otherwise, create a new feature matrix
    X = df[numerical_features + categorical_features]
    y = df['Survived'] if 'Survived' in df.columns else None
    
    # Create preprocessor
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit preprocessor on data and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Save preprocessor for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, os.path.join('models', 'preprocessor.pkl'))
    
    # Also save processed data for later analysis
    if y is not None:
        processed_data = {
            'X_processed': X_processed,
            'X_train': X,
            'X_test': None,
            'y_train': y,
            'y_test': None
        }
        joblib.dump(processed_data, os.path.join('models', 'processed_data.pkl'))
    
    return X_processed, preprocessor

def main():
    """Main function to run preprocessing."""
    # Load data
    data_path = os.path.join('data', 'titanic.csv')
    df = load_data(data_path)
    
    # Display data info
    print("\nDataset info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Save preprocessor and data splits
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, os.path.join('models', 'preprocessor.pkl'))
    
    # Save data splits
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    joblib.dump(processed_data, os.path.join('models', 'processed_data.pkl'))
    
    print("\nPreprocessing complete! Data splits and preprocessor saved.")

if __name__ == "__main__":
    main() 