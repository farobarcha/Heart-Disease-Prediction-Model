#!/usr/bin/env python
# coding: utf-8

# # Part 3: Model Saving and Loading

# In[63]:


# import Joblib
import joblib


# ### Model Saving and Loading Methods

# In[65]:


# Function to save the trained model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Function to load the saved model
def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model


# ### Save Model

# In[67]:


# Save the model
save_model(rf_model, "random_forest_model.pkl")


# ### Load Model

# In[69]:


loaded_model = load_model('random_forest_model.pkl')


# In[73]:


loaded_model.predict(X_test)

