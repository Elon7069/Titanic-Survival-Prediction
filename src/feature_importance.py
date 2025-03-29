import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def load_preprocessed_data():
    """Load preprocessed data splits."""
    print("Loading preprocessed data...")
    data_path = os.path.join('models', 'processed_data.pkl')
    preprocessor_path = os.path.join('models', 'preprocessor.pkl')
    
    processed_data = joblib.load(data_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return processed_data, preprocessor

def get_feature_names(preprocessor):
    """Get feature names after preprocessing."""
    # Extract column names for categorical features
    cat_features = preprocessor.transformers[1][2]
    cat_encoder = preprocessor.transformers[1][1].named_steps['onehot']
    cat_feature_names = []
    for i, feature in enumerate(cat_features):
        categories = cat_encoder.categories[i]
        for category in categories:
            cat_feature_names.append(f"{feature}_{category}")
    
    # Extract column names for numerical features
    num_features = preprocessor.transformers[0][2]
    
    # Combine all feature names
    feature_names = num_features + cat_feature_names
    
    return feature_names

def analyze_random_forest_importance(X_train, y_train, X_test, feature_names):
    """Analyze feature importance using Random Forest."""
    print("\nAnalyzing feature importance using Random Forest...")
    
    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Random Forest Feature Importances')
    plt.bar(range(X_train.shape[1]), importances[indices],
           color='royalblue', align='center')
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'rf_feature_importance.png'))
    
    # Print feature importances
    print("\nFeature importances:")
    for i in range(len(indices)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return rf, importances, indices

def analyze_permutation_importance(model, X_test, y_test, feature_names):
    """Analyze feature importance using permutation importance."""
    print("\nAnalyzing permutation importance...")
    
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Sort permutation importance
    perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
    
    # Plot permutation importance
    plt.figure(figsize=(12, 8))
    plt.title('Permutation Feature Importances')
    plt.bar(range(X_test.shape[1]), perm_importance.importances_mean[perm_indices],
           color='lightblue', align='center', yerr=perm_importance.importances_std[perm_indices])
    plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in perm_indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'permutation_importance.png'))
    
    # Print permutation importances
    print("\nPermutation importances:")
    for i in range(len(perm_indices)):
        print(f"{feature_names[perm_indices[i]]}: {perm_importance.importances_mean[perm_indices[i]]:.4f} ± {perm_importance.importances_std[perm_indices[i]]:.4f}")
    
    return perm_importance, perm_indices

def main():
    """Main function to analyze feature importance."""
    # Load preprocessed data
    processed_data, preprocessor = load_preprocessed_data()
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_test = processed_data['y_test']
    
    # Get feature names
    feature_names = get_feature_names(preprocessor)
    
    # Train model and get feature importances
    rf, importances, indices = analyze_random_forest_importance(
        preprocessor.transform(X_train), y_train, 
        preprocessor.transform(X_test), feature_names
    )
    
    # Get permutation importances
    perm_importance, perm_indices = analyze_permutation_importance(
        rf, preprocessor.transform(X_test), y_test, feature_names
    )
    
    # Visualize top features with partial dependence plots
    print("\nFeature importance analysis complete!")

if __name__ == "__main__":
    main() 