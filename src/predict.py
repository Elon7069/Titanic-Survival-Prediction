import pandas as pd
import numpy as np
import joblib
import os

def load_model():
    """Load the trained model and model info."""
    print("Loading model...")
    
    # Load model info
    model_info_path = os.path.join('models', 'model_info.pkl')
    model_info = joblib.load(model_info_path)
    
    # Load best model
    model_path = os.path.join('models', f"{model_info['model_name']}.pkl")
    model = joblib.load(model_path)
    
    print(f"Loaded {model_info['model_name']} with accuracy: {model_info['accuracy']:.4f}")
    
    return model, model_info

def make_sample_prediction(model):
    """Make a prediction on a sample passenger."""
    print("\nMaking prediction for a sample passenger...")
    
    # Create sample passengers
    sample_passengers = [
        {
            'PassengerId': 1000,
            'Pclass': 1,
            'Name': 'Johnson, Mr. William',
            'Sex': 'male',
            'Age': 35,
            'SibSp': 0,
            'Parch': 0,
            'Ticket': 'PC 17755',
            'Fare': 71.2833,
            'Cabin': 'C85',
            'Embarked': 'C'
        },
        {
            'PassengerId': 1001,
            'Pclass': 3,
            'Name': 'Smith, Mrs. Jane',
            'Sex': 'female',
            'Age': 22,
            'SibSp': 1,
            'Parch': 1,
            'Ticket': 'A/5 21171',
            'Fare': 7.25,
            'Cabin': '',
            'Embarked': 'S'
        }
    ]
    
    # Convert to DataFrame
    sample_df = pd.DataFrame(sample_passengers)
    
    # Preprocess sample data
    from preprocess import preprocess_data
    X_sample, _ = preprocess_data(sample_df)
    
    # Make predictions
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)
    
    # Display results
    for i, passenger in enumerate(sample_passengers):
        survival = "Survived" if predictions[i] == 1 else "Did Not Survive"
        print(f"\nPassenger: {passenger['Name']}")
        print(f"Class: {passenger['Pclass']}, Sex: {passenger['Sex']}, Age: {passenger['Age']}")
        print(f"Prediction: {survival} (Probability: {probabilities[i][1]:.4f})")
    
    return predictions, probabilities

def predict_new_data(model, data_path=None):
    """Make predictions on new data."""
    if data_path is None:
        print("\nNo new data provided. Using sample passengers instead.")
        return make_sample_prediction(model)
    
    print(f"\nMaking predictions for data from {data_path}...")
    
    # Load new data
    new_data = pd.read_csv(data_path)
    
    # Preprocess new data
    from preprocess import preprocess_data
    X_new, _ = preprocess_data(new_data)
    
    # Make predictions
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'PassengerId': new_data['PassengerId'] if 'PassengerId' in new_data.columns else range(len(new_data)),
        'Name': new_data['Name'] if 'Name' in new_data.columns else ['Passenger ' + str(i) for i in range(len(new_data))],
        'Prediction': predictions,
        'Survival': ['Survived' if p == 1 else 'Did Not Survive' for p in predictions],
        'Probability': probabilities[:, 1]
    })
    
    # Save results
    results_path = os.path.join('data', 'predictions.csv')
    results.to_csv(results_path, index=False)
    print(f"Predictions saved to {results_path}")
    
    return results

def main():
    """Main function to run predictions."""
    # Load model
    model, model_info = load_model()
    
    # Check if test data exists
    test_data_path = os.path.join('data', 'test.csv')
    if os.path.exists(test_data_path):
        results = predict_new_data(model, test_data_path)
        print(f"\nPredictions made for {len(results)} passengers in the test set.")
    else:
        predictions, probabilities = make_sample_prediction(model)
        
    print("\nPrediction process complete!")

if __name__ == "__main__":
    main() 