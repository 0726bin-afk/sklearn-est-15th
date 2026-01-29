import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('titanic_voting_model.pkl')

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Predicts survival based on passenger information.
    """
    # Create a DataFrame from the inputs
    data = {
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    }
    df = pd.DataFrame(data)
    
    # Predict probabilities (since VotingClassifier was soft)
    try:
        proba = model.predict_proba(df)[0]
        prediction = model.predict(df)[0]
        
        result = "Survived" if prediction == 1 else "Did not survive"
        prob_survived = proba[1]
        
        return f"{result} (Probability of Survival: {prob_survived:.2%})"
    except Exception as e:
        return f"Error: {str(e)}"

# Define Gradio Interface
iface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown([1, 2, 3], label="Passenger Class (Pclass)"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Slider(0, 100, step=1, label="Age"),
        gr.Number(label="Siblings/Spouses Aboard (SibSp)", value=0),
        gr.Number(label="Parents/Children Aboard (Parch)", value=0),
        gr.Number(label="Fare", value=32.2),
        gr.Dropdown(["S", "C", "Q"], label="Port of Embarkation (Embarked)")
    ],
    outputs="text",
    title="Titanic Survivor Prediction",
    description="Enter passenger details to predict if they would have survived the Titanic disaster."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")
