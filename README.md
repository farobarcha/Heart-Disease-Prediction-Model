# Heart Disease Prediction Model

The project consists of five parts:
1. Data Cleaning
2. Model Building
3. Model Saving and Loading
4. FastAPI Endpoint for the Model
5. Deployment on Hugging Face with Gradio UI

## Prerequisites

Make sure you have the following libraries installed:
```bash
pip install pandas scikit-learn joblib fastapi uvicorn gradio nest_asyncio pyngrok
```

You also need:
- **Jupyter Notebook** or any Python IDE.
- **ngrok** for exposing local servers.
- **Gradio** for creating a simple UI.

## Steps

### Part 1: Data Cleaning

1. **Load the Dataset**: We load the dataset from `heart.csv`.
2. **Check for Missing Values**: No missing values in this dataset.
3. **Label Encoding**: Encode categorical columns (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) using `LabelEncoder`.

```python
# Load and clean data
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("heart.csv")

categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
encoder = LabelEncoder()

def encode_columns(df, columns):
    for column in columns:
        df[column] = encoder.fit_transform(df[column])
    return df

cleaned_data = encode_columns(data, categorical_cols)
cleaned_data.to_csv('cleaned_heart_disease_data.csv', index=False)
```

### Part 2: Model Building

1. **Train/Test Split**: Split the cleaned data into training and testing sets (80% train, 20% test).
2. **Model Selection and Training**: Use a Random Forest classifier.
3. **Evaluation**: Evaluate the model using accuracy, precision, recall, and F1 score.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X = cleaned_data.drop('HeartDisease', axis=1)
y = cleaned_data['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
rf_y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)
rf_f1 = f1_score(y_test, rf_y_pred)

print(f"Accuracy: {rf_accuracy:.2f}")
print(f"Precision: {rf_precision:.2f}")
print(f"Recall: {rf_recall:.2f}")
print(f"F1 Score: {rf_f1:.2f}")
```

### Part 3: Model Saving and Loading

1. **Save the Model**: Save the trained Random Forest model using `joblib`.
2. **Load the Model**: Reload the saved model for later use.

```python
import joblib

# Save model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

save_model(rf_model, "random_forest_model.pkl")

# Load model
def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model

loaded_model = load_model('random_forest_model.pkl')
```

### Part 4: FastAPI Endpoint for the Model

1. **Create a FastAPI Application**: Define a prediction endpoint that takes input data and returns predictions.
2. **Run the API Locally**: Use `uvicorn` to run the FastAPI app.
3. **Expose API to Public**: Use `ngrok` to create a public link to the FastAPI app.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load("random_forest_model.pkl")

class HeartDiseaseInput(BaseModel):
    Age: int
    Sex: int
    ChestPainType: int
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: int
    MaxHR: int
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int

@app.post("/predict")
async def predict(input_data: HeartDiseaseInput):
    input_array = np.array([[input_data.Age, input_data.Sex, input_data.ChestPainType,
                             input_data.RestingBP, input_data.Cholesterol, input_data.FastingBS,
                             input_data.RestingECG, input_data.MaxHR, input_data.ExerciseAngina,
                             input_data.Oldpeak, input_data.ST_Slope]])
    prediction = model.predict(input_array)
    return {"prediction": int(prediction[0])}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

To expose the API to the public, use `ngrok`:
```bash
ngrok http 8000
```

### Part 5: Deployment on Hugging Face with Gradio UI

1. **Gradio Interface**: Create a simple UI with Gradio for users to interact with the model.
2. **Deploy on Hugging Face Spaces**: Deploy the app on Hugging Face Spaces.

```python
import gradio as gr
import joblib

# Load model
model = joblib.load('random_forest_model.pkl')

# Prediction function
def predict_heart_disease(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    prediction = model.predict([[int(Age), int(Sex), int(ChestPainType), int(RestingBP), int(Cholesterol), 
                                 int(FastingBS), int(RestingECG), int(MaxHR), int(ExerciseAngina), 
                                 float(Oldpeak), int(ST_Slope)]])
    return "Heart Disease" if prediction == 1 else "No Heart Disease"

# Gradio interface
iface = gr.Interface(
    fn=predict_heart_disease, 
    inputs=[
        gr.Number(label="Age"),
        gr.Radio([0, 1], label="Sex (0 = Female, 1 = Male)"),
        gr.Radio([0, 1, 2, 3], label="ChestPainType (0 = ASY, 1 = ATA, 2 = NAP, 3 = TA)"),
        gr.Number(label="RestingBP"),
        gr.Number(label="Cholesterol"),
        gr.Radio([0, 1], label="FastingBS (0 = No, 1 = Yes)"),
        gr.Radio([0, 1, 2], label="RestingECG (0 = Normal, 1 = LVH, 2 = ST)"),
        gr.Number(label="MaxHR"),
        gr.Radio([0, 1], label="ExerciseAngina (0 = No, 1 = Yes)"),
        gr.Number(label="Oldpeak"),
        gr.Radio([0, 1, 2], label="ST_Slope (0 = Up, 1 = Flat, 2 = Down)")
    ],
    outputs=gr.Textbox(),
    title="Heart Disease Prediction",
    description="Predicts whether a person has heart disease based on medical data."
)

iface.launch()
```

### Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/farobarcha/Heart-Disease-Prediction-Model.git
   cd Heart-Disease-Prediction-Model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run FastAPI locally:
   ```bash
   uvicorn app:app --reload
   ```

4. For Hugging Face deployment, push the updated `app.py` and `requirements.txt` to your Hugging Face Space.