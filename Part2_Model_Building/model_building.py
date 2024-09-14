#!/usr/bin/env python
# coding: utf-8

# # Part 2: Model Building

# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ### Split into Train/Test sets

# In[45]:


# Split dataset into test/train
X = cleaned_data.drop('HeartDisease', axis=1)
y = cleaned_data['HeartDisease']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Model Selection

# In[49]:


# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)


# ### Model Training

# In[51]:


# Train the model
rf_model.fit(X_train, y_train)


# ### Model Score

# In[55]:


result = rf_model.score(X_test, y_test)
print(result)


# ### Predict Results

# In[57]:


# Make predictions on the test set
rf_y_pred = rf_model.predict(X_test)


# ### Evaluation

# In[61]:


# Evaluate the model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_precision = precision_score(y_test, rf_y_pred)
rf_recall = recall_score(y_test, rf_y_pred)
rf_f1 = f1_score(y_test, rf_y_pred)

# Print the evaluation metrics
print("\nRandom Forest:")
print(f"Accuracy: {rf_accuracy:.2f}")
print(f"Precision: {rf_precision:.2f}")
print(f"Recall: {rf_recall:.2f}")
print(f"F1 Score: {rf_f1:.2f}")

