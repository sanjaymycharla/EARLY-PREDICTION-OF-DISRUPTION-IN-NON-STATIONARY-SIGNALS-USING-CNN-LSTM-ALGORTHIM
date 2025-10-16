#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("disruption_ttd_model.h5")

# Load the unseen dataset
unseen_data = pd.read_csv("C:\\Users\\mycha\\OneDrive\\Documents\\my final project\\allfiles33\\csv\\test\\final_aligned_testdataset_fixed37.csv")  # Replace with actual path

# Extract the required columns, including 'Plasmact'
selected_columns = ["Plasmact","H-alpha", "Mirnov", "DeltaR", "SXR"]
unseen_features = unseen_data[selected_columns].values

# Normalize each feature separately (same approach as training)
X_unseen_scaled = np.zeros_like(unseen_features)  # Preserve shape
for i in range(unseen_features.shape[1]):
    X_unseen_scaled[:, i] = (unseen_features[:, i] - unseen_features[:, i].min()) / \
                            (unseen_features[:, i].max() - unseen_features[:, i].min())

# Reshape for CNN-LSTM input
X_unseen_scaled = X_unseen_scaled.reshape((X_unseen_scaled.shape[0], X_unseen_scaled.shape[1], 1))

# Function to create sliding windows (same as training)
def create_sequences(data, window_size=30):
    X_seq = []
    for i in range(len(data) - window_size):
        X_seq.append(data[i:i+window_size])
    return np.array(X_seq)

# Create sequences for prediction
window_size = 30  # Must match training setup
X_unseen_seq = create_sequences(X_unseen_scaled, window_size)

# Predict TTD using the trained model
y_pred_unseen = model.predict(X_unseen_seq)

# Extract the time column for reference
time_values = unseen_data["Time"].values[window_size:]  # Align with sequences

# Find the first predicted disruption time (lowest predicted TTD)
predicted_disruption_index = np.argmin(y_pred_unseen)  # Smallest TTD means closest disruption
predicted_disruption_time = time_values[predicted_disruption_index]

# Calculate the Predicted Time to Disruption (TTD)
TTD_predicted = predicted_disruption_time - time_values[0]

print(f"Predicted Disruption Time: {predicted_disruption_time:.3f} ms")
print(f"Predicted TTD: {TTD_predicted:.3f} ms")

# Plot Predicted vs Time
plt.figure(figsize=(10, 5))
plt.plot(time_values, y_pred_unseen, label="Predicted TTD", color='red')
plt.axvline(predicted_disruption_time, color='blue', linestyle="--", label="Predicted Disruption Time")
plt.xlabel("Time (ms)")
plt.ylabel("Predicted TTD (ms)")
plt.title("Predicted Time to Disruption (TTD) Over Time")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




