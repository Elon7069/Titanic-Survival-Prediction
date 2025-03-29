import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data():
    """Load the Titanic dataset."""
    data_path = os.path.join('data', 'titanic.csv')
    print(f"Loading data from {data_path}...")
    return pd.read_csv(data_path)

def create_plots(df):
    """Create visualizations for the Titanic dataset."""
    print("Creating visualizations...")
    
    # Create directory for plots
    os.makedirs('data/plots', exist_ok=True)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # 1. Survival rate visualization
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Survived', data=df, palette='Blues')
    plt.title('Survival Distribution')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.savefig(os.path.join('data/plots', '1_survival_distribution.png'))
    plt.close()
    
    # 2. Survival by gender
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sex', hue='Survived', data=df, palette='Blues')
    plt.title('Survival by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.savefig(os.path.join('data/plots', '2_survival_by_gender.png'))
    plt.close()
    
    # 3. Survival by passenger class
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='Blues')
    plt.title('Survival by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join('data/plots', '3_survival_by_class.png'))
    plt.close()
    
    # 4. Age distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30, palette='Blues')
    plt.title('Age Distribution by Survival')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig(os.path.join('data/plots', '4_age_distribution.png'))
    plt.close()
    
    # 5. Fare distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Fare', hue='Survived', multiple='stack', bins=30, palette='Blues')
    plt.title('Fare Distribution by Survival')
    plt.xlabel('Fare')
    plt.ylabel('Count')
    plt.savefig(os.path.join('data/plots', '5_fare_distribution.png'))
    plt.close()
    
    # 6. Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='Blues', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join('data/plots', '6_correlation_heatmap.png'))
    plt.close()
    
    # 7. Embarked port and survival
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Embarked', hue='Survived', data=df, palette='Blues')
    plt.title('Survival by Embarked Port')
    plt.xlabel('Embarked Port')
    plt.ylabel('Count')
    plt.savefig(os.path.join('data/plots', '7_survival_by_port.png'))
    plt.close()
    
    # 8. Family size and survival
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    plt.figure(figsize=(12, 6))
    sns.countplot(x='FamilySize', hue='Survived', data=df, palette='Blues')
    plt.title('Survival by Family Size')
    plt.xlabel('Family Size')
    plt.ylabel('Count')
    plt.savefig(os.path.join('data/plots', '8_survival_by_family.png'))
    plt.close()
    
    print(f"All plots saved to data/plots directory.")

def main():
    """Main function to run visualizations."""
    # Load data
    df = load_data()
    
    # Create visualizations
    create_plots(df)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main() 