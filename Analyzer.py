# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def create_lung_cancer_predictor():
    # Load the dataset
    data = pd.read_csv('Lung_Cancer.csv')
    
    # Clean column names (remove trailing spaces)
    data.columns = data.columns.str.strip()
    
    # Encode the target variable
    le = LabelEncoder()
    data['LUNG_CANCER'] = le.fit_transform(data['LUNG_CANCER'])
    
    # Encode the 'GENDER' column
    data['GENDER'] = data['GENDER'].map({'M': 0, 'F': 1})
    
    # Separate features and target
    X = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']
    
    # Train the Naive Bayes model on the full dataset
    nb_model = GaussianNB()
    nb_model.fit(X, y)
    
    # Train a Random Forest model to get feature importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # Get feature importance
    feature_importance = {}
    for i, feature in enumerate(X.columns):
        feature_importance[feature] = rf_model.feature_importances_[i]
    
    # Normalize feature importance to sum to 1
    total_importance = sum(feature_importance.values())
    for feature in feature_importance:
        feature_importance[feature] /= total_importance
    
    return nb_model, X.columns, le.classes_, feature_importance

def get_user_input(feature_names, feature_importance):
    """Get user input for all features and display importance weights"""
    user_data = {}
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeature Importance (Higher is more influential):")
    print("-" * 50)
    for feature, importance in sorted_features:
        if feature in ['GENDER', 'AGE']:
            continue  # Skip displaying weights for demographic features
        print(f"{feature.title().replace('_', ' ')}: {importance:.3f} ({importance*100:.1f}%)")
    print("-" * 50)
    
    # Gender input
    while True:
        gender = input("\nWhat is your gender? (M/F): ").upper()
        if gender in ['M', 'F']:
            user_data['GENDER'] = gender
            break
        else:
            print("Please enter 'M' for male or 'F' for female.")
    
    # Age input
    while True:
        try:
            age = int(input("What is your age? (in years): "))
            if 0 < age < 120:
                user_data['AGE'] = age
                break
            else:
                print("Please enter a valid age between 1 and 120.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Input for binary features (1 or 2), ordered by importance
    binary_features = [f for f, _ in sorted_features if f not in ['GENDER', 'AGE']]
    
    print("\nFor the following questions, enter:")
    print("1 for NO or LOW")
    print("2 for YES or HIGH")
    print()
    
    for feature in binary_features:
        importance = feature_importance[feature]
        while True:
            try:
                response = int(input(f"Do you have {feature.title().replace('_', ' ')}? (1=No/Low, 2=Yes/High) "))
                if response in [1, 2]:
                    user_data[feature] = response
                    break
                else:
                    print("Please enter 1 (No/Low) or 2 (Yes/High).")
            except ValueError:
                print("Please enter a valid number (1 or 2).")
    
    return user_data

def predict_lung_cancer_with_weights(model, feature_names, class_names, user_data, feature_weights):
    """Predict lung cancer risk based on user input and feature weights"""
    # Convert gender to numeric
    if 'GENDER' in user_data:
        user_data['GENDER'] = 0 if user_data['GENDER'] == 'M' else 1
    
    # Convert to DataFrame with the same columns as training data
    user_df = pd.DataFrame([user_data], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(user_df)[0]
    
    # Get probability
    probabilities = model.predict_proba(user_df)[0]
    
    # Calculate weighted risk score (for additional context)
    risk_score = 0
    for feature in user_data:
        if feature == 'AGE':
            # Normalize age between 0 and 1 (assuming max age 100)
            normalized_value = min(user_data[feature] / 100, 1.0)
        elif feature == 'GENDER':
            normalized_value = user_data[feature]  # Already 0 or 1
        else:
            # Convert 1,2 scale to 0,1 scale
            normalized_value = (user_data[feature] - 1) / 1
        
        risk_score += normalized_value * feature_weights[feature]
    
    # Scale risk score to 0-100
    risk_score = risk_score * 100
    
    # Return result
    result = {
        'prediction': class_names[prediction],
        'probability': probabilities[prediction] * 100,  # Convert to percentage
        'risk_score': risk_score,
        'feature_contribution': {}
    }
    
    # Calculate contribution of each feature to the risk score
    for feature in user_data:
        if feature == 'AGE':
            normalized_value = min(user_data[feature] / 100, 1.0)
        elif feature == 'GENDER':
            normalized_value = user_data[feature]
        else:
            normalized_value = (user_data[feature] - 1) / 1
        
        contribution = normalized_value * feature_weights[feature] * 100
        result['feature_contribution'][feature] = contribution
    
    return result

def main():
    print("Lung Cancer Risk Assessment Tool")
    print("=" * 35)
    print("This tool helps assess your risk of lung cancer based on various factors.")
    print("Each factor has a different weight in the prediction based on its importance.")
    
    # Create the predictor model and get feature importance
    model, feature_names, class_names, feature_weights = create_lung_cancer_predictor()
    
    # Get user input
    user_data = get_user_input(feature_names, feature_weights)
    
    # Make prediction with weights
    result = predict_lung_cancer_with_weights(model, feature_names, class_names, user_data, feature_weights)
    
    # Display result
    print("\nLung Cancer Risk Assessment Result:")
    print("-" * 50)
    if result['prediction'] == 'YES':
        print(f"Risk Assessment: POTENTIAL RISK DETECTED")
        print(f"The model indicates a {result['probability']:.1f}% chance of lung cancer risk.")
        print(f"Weighted risk score: {result['risk_score']:.1f}/100")
    else:
        print(f"Risk Assessment: LOW RISK")
        print(f"The model indicates a {100-result['probability']:.1f}% chance of no lung cancer.")
        print(f"Weighted risk score: {result['risk_score']:.1f}/100")
    
    # Display top contributing factors
    print("\nTop Contributing Factors:")
    contributions = sorted(result['feature_contribution'].items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, contribution in contributions[:5]:  # Show top 5 contributors
        print(f"- {feature.title().replace('_', ' ')}: {contribution:.1f}")
    
    print("\nIMPORTANT: This is NOT a diagnosis. This is only a risk assessment tool.")
    print("Please consult a healthcare professional for proper medical evaluation.")

if __name__ == "__main__":
    main()
