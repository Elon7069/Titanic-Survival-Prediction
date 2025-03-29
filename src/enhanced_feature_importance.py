import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.preprocessing import StandardScaler
import re

def extract_title(name):
    """Extract title from name."""
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def engineer_features(df):
    """Engineer new features from existing data."""
    print("Engineering new features...")
    
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
        df['FareGroup'] = pd.qcut(df['Fare'], 
                                q=4,
                                labels=['Low', 'Medium', 'High', 'Very High'],
                                duplicates='drop')
    else:
        # For single passengers or when all fares are identical
        # Determine the fare group based on the overall fare distribution from the training data
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

def analyze_feature_importance(X_train, y_train, X_test, y_test, feature_names):
    """Analyze feature importance using multiple methods."""
    print("\nAnalyzing feature importance...")
    
    # Convert feature_names to list if it's pandas Index
    feature_names = list(feature_names)
    
    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Random Forest Feature Importances
    plt.subplot(2, 2, 1)
    plt.title('Random Forest Feature Importances')
    bars = plt.bar(range(X_train.shape[1]), importances[indices],
                  color='royalblue', align='center')
    plt.xticks(range(X_train.shape[1]), 
               [feature_names[i] for i in indices], 
               rotation=90)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # 2. Permutation Importance
    plt.subplot(2, 2, 2)
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    perm_indices = np.argsort(perm_importance.importances_mean)[::-1]
    
    plt.title('Permutation Feature Importances')
    bars = plt.bar(range(X_test.shape[1]), 
                  perm_importance.importances_mean[perm_indices],
                  color='lightblue', 
                  align='center',
                  yerr=perm_importance.importances_std[perm_indices])
    
    plt.xticks(range(X_test.shape[1]), 
               [feature_names[i] for i in perm_indices], 
               rotation=90)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # 3. Feature Correlation with Target
    plt.subplot(2, 2, 3)
    correlations = []
    for i in range(X_train.shape[1]):
        correlation = np.corrcoef(X_train[:, i], y_train)[0, 1]
        correlations.append(abs(correlation))
    
    corr_indices = np.argsort(correlations)[::-1]
    plt.title('Feature Correlation with Target')
    bars = plt.bar(range(X_train.shape[1]), 
                  [correlations[i] for i in corr_indices],
                  color='lightgreen', 
                  align='center')
    
    plt.xticks(range(X_train.shape[1]), 
               [feature_names[i] for i in corr_indices], 
               rotation=90)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # 4. Feature Distribution by Target
    plt.subplot(2, 2, 4)
    # Select top 5 most important features
    top_features = [feature_names[i] for i in indices[:5]]
    plt.title('Top 5 Features Distribution by Target')
    
    # Create box plots for top features
    data_to_plot = []
    labels = []
    for feature in top_features:
        feature_idx = feature_names.index(feature)
        data_to_plot.append(X_train[:, feature_idx])
        labels.append(feature)
    
    plt.boxplot(data_to_plot, labels=labels, vert=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'enhanced_feature_importance.png'))
    
    # Print feature importance summary
    print("\nFeature Importance Summary:")
    print("=" * 50)
    print("Random Forest Importance:")
    for i in range(len(indices)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    print("\nPermutation Importance:")
    for i in range(len(perm_indices)):
        print(f"{feature_names[perm_indices[i]]}: {perm_importance.importances_mean[perm_indices[i]]:.4f} ± {perm_importance.importances_std[perm_indices[i]]:.4f}")
    
    print("\nCorrelation with Target:")
    for i in range(len(corr_indices)):
        print(f"{feature_names[corr_indices[i]]}: {correlations[corr_indices[i]]:.4f}")
    
    return rf, importances, indices, perm_importance, perm_indices, correlations, corr_indices

def main():
    """Main function to analyze feature importance."""
    # Load data
    data_path = os.path.join('data', 'titanic.csv')
    df = pd.read_csv(data_path)
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare features for modeling
    categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'CabinDeck']
    numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 
                         'IsAlone', 'FarePerPerson', 'HasCabin', 'TicketNumber']
    
    # Create feature matrix
    X = pd.get_dummies(df[categorical_features], drop_first=True)
    X = pd.concat([X, df[numerical_features]], axis=1)
    y = df['Survived']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Analyze feature importance
    rf, importances, indices, perm_importance, perm_indices, correlations, corr_indices = analyze_feature_importance(
        X_train_scaled, y_train, X_test_scaled, y_test, X.columns
    )
    
    print("\nFeature importance analysis complete!")

if __name__ == "__main__":
    main() 