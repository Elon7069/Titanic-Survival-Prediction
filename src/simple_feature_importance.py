import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def load_data():
    """Load the Titanic dataset and preprocess it directly."""
    print("Loading data...")
    data_path = os.path.join('data', 'titanic.csv')
    df = pd.read_csv(data_path)
    
    # Basic preprocessing
    # Extract relevant features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    y = df['Survived']
    
    # Handle missing values
    X['Age'].fillna(X['Age'].median(), inplace=True)
    X['Fare'].fillna(X['Fare'].median(), inplace=True)
    
    # Encode categorical variables
    X = pd.get_dummies(X, columns=['Sex'], drop_first=False)
    
    return X, y, features

def analyze_feature_importance(X, y):
    """Analyze feature importance using Random Forest."""
    print("\nAnalyzing feature importance...")
    
    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_names = X.columns
    
    # Sort importances
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importance for Titanic Survival Prediction')
    plt.bar(range(X.shape[1]), sorted_importances, align='center')
    plt.xticks(range(X.shape[1]), sorted_feature_names, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'feature_importance.png'))
    
    # Print feature importances
    print("\nFeature importances:")
    for i in range(len(indices)):
        print(f"{sorted_feature_names[i]}: {sorted_importances[i]:.4f}")
    
    return rf, sorted_feature_names, sorted_importances

def main():
    """Main function to analyze feature importance."""
    # Load and preprocess data
    X, y, original_features = load_data()
    
    # Analyze feature importance
    rf, top_features, importances = analyze_feature_importance(X, y)
    
    print("\nFeature importance analysis complete!")
    print(f"\nTop 3 most important features:")
    for i in range(min(3, len(top_features))):
        print(f"{i+1}. {top_features[i]} (importance: {importances[i]:.4f})")

if __name__ == "__main__":
    main() 