#!/usr/bin/env python
# coding: utf-8

# # Part Part 5: Deployment at Hugging Face with a Simple UI using Gradio or Streamlit

# In[40]:


pip install gradio


# In[65]:


import gradio as gr
import joblib

# Load model
model = joblib.load('random_forest_model.pkl')

# Define a function to make predictions
def predict_heart_disease(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    # Make a prediction using the loaded model
    prediction = model.predict([[int(Age), int(Sex), int(ChestPainType), int(RestingBP), int(Cholesterol), 
                                 int(FastingBS), int(RestingECG), int(MaxHR), int(ExerciseAngina), 
                                 float(Oldpeak), int(ST_Slope)]])
    
    # Interpret the prediction result
    if prediction == 1:
        return "Heart Disease"
    else:
        return "No Heart Disease"

# Create Gradio Interface using the appropriate components
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

# Launch the interface
iface.launch()

