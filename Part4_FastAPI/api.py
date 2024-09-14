#!/usr/bin/env python
# coding: utf-8

# # Part 4: FastAPI Endpoint for the Model

# In[134]:


from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np


# In[136]:


# Initialize FastAPI app
app = FastAPI()

# Load the trained model (saved as random_forest_model.pkl)
model = joblib.load("random_forest_model.pkl")

# Define the input data model for the API
class HeartDiseaseInput(BaseModel):
    Age: int
    Sex: int  # 0 or 1
    ChestPainType: int  # Encoded as integers (example: 0=ASY, 1=ATA, etc.)
    RestingBP: int
    Cholesterol: int
    FastingBS: int  # 0 or 1
    RestingECG: int  # Encoded as integers (example: 0=Normal, 1=LVH, etc.)
    MaxHR: int
    ExerciseAngina: int  # 0 or 1
    Oldpeak: float
    ST_Slope: int  # Encoded as integers (example: 0=Up, 1=Flat, etc.)

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: HeartDiseaseInput):
    # Convert the input data into a NumPy array for prediction
    input_array = np.array([[input_data.Age, input_data.Sex, input_data.ChestPainType,
                             input_data.RestingBP, input_data.Cholesterol, input_data.FastingBS,
                             input_data.RestingECG, input_data.MaxHR, input_data.ExerciseAngina,
                             input_data.Oldpeak, input_data.ST_Slope]])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_array)

    # Return the prediction result
    return {"prediction": int(prediction[0])}


# In[114]:


get_ipython().system('ngrok config add-authtoken 2m44emMfK1DXJrzqubTxplmJH43_5RvRvJGUzrdidqMs59xZi')


# In[ ]:


import uvicorn
import nest_asyncio
from pyngrok import ngrok

ngrok_tunnel = ngrok.connect(8000)
print('Public URL: ', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)