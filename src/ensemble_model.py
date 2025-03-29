import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import re

# Import feature engineering function from enhanced_feature_importance.py
from enhanced_feature_importance import engineer_features

def train_ensemble_model():
    """Train an ensemble model using multiple algorithms."""
    print("Training ensemble model...")
    
    # Load data
    data_path = os.path.join('data', 'titanic.csv')
    df = pd.read_csv(data_path)
    
    # Fill missing values before feature engineering (avoiding inplace warnings)
    df = df.copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create base models
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    lr = LogisticRegression(C=1.0, random_state=42)
    
    # Create voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('svc', svc),
            ('lr', lr)
        ],
        voting='soft'  # Use predicted probabilities for voting
    )
    
    # Train ensemble model
    ensemble.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = ensemble.predict(X_test_scaled)
    y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble model accuracy: {accuracy:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Did Not Survive', 'Survived'],
               yticklabels=['Did Not Survive', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Ensemble (Accuracy: {accuracy:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'confusion_matrix_ensemble.png'))
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Ensemble Model')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'roc_curve_ensemble.png'))
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Analyze individual model performances
    print("\nIndividual Model Performances:")
    models = {
        'Random Forest': rf,
        'Gradient Boosting': gb,
        'SVC': svc,
        'Logistic Regression': lr,
        'Ensemble': ensemble
    }
    
    accuracy_scores = {}
    
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_model_pred = model.predict(X_test_scaled)
        model_accuracy = accuracy_score(y_test, y_model_pred)
        accuracy_scores[name] = model_accuracy
        print(f"{name} accuracy: {model_accuracy:.4f}")
        
        if name != 'Ensemble':
            y_model_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr_model, tpr_model, _ = roc_curve(y_test, y_model_prob)
            roc_auc_model = auc(fpr_model, tpr_model)
            plt.plot(fpr_model, tpr_model, lw=2, 
                   label=f'{name} (AUC = {roc_auc_model:.4f})')
    
    # Add ensemble ROC to the comparison
    plt.plot(fpr, tpr, color='red', lw=2, 
            label=f'Ensemble (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'roc_curve_comparison.png'))
    
    # Create accuracy comparison bar chart
    plt.figure(figsize=(12, 6))
    models_list = list(accuracy_scores.keys())
    accuracy_list = [accuracy_scores[model] for model in models_list]
    
    bars = plt.bar(models_list, accuracy_list, color=['lightblue', 'lightgreen', 'lightsalmon', 'wheat', 'red'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.ylim(0.7, 0.9)  # Adjust as needed
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join('models', 'model_accuracy_comparison.png'))
    
    # Save models and metadata
    model_info = {
        'model_name': 'Ensemble',
        'accuracy': accuracy,
        'components': ['RandomForest', 'GradientBoosting', 'SVC', 'LogisticRegression'],
        'feature_names': list(X.columns),
        'parameters': {
            'rf_n_estimators': 200,
            'rf_max_depth': 10,
            'gb_n_estimators': 200,
            'gb_learning_rate': 0.1,
            'svc_kernel': 'rbf',
            'lr_C': 1.0,
            'voting': 'soft'
        }
    }
    
    # Save ensemble model
    joblib.dump(ensemble, os.path.join('models', 'Ensemble_advanced.pkl'))
    
    # Save preprocessor information
    preprocessor_info = {
        'scaler': scaler,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features
    }
    joblib.dump(preprocessor_info, os.path.join('models', 'ensemble_preprocessor.pkl'))
    
    # Save model info
    joblib.dump(model_info, os.path.join('models', 'advanced_model_info.pkl'))
    
    return ensemble, model_info, preprocessor_info

if __name__ == "__main__":
    train_ensemble_model() 