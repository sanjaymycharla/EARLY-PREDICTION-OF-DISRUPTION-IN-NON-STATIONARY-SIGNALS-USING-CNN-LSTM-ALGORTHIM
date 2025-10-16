#!/usr/bin/env python
# coding: utf-8

# In[1]:


#step-1 import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[2]:


#step-2 :import the dataset
df = pd.read_csv("C:\\Users\\mycha\\Downloads\\labeled_dataset_with_TTD.csv")


# In[3]:


# step 3:Check the structure of the dataset
print(df.head())
print(df.info())


# In[4]:


# step-4:Separate features and target variable
X = df[['Plasmact', 'H-alpha', 'Mirnov', 'DeltaR', 'SXR']].values
y = df['TTD'].values  


# In[5]:


# Step-5:Normalize each feature separately using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = np.zeros_like(X)  # Preserve shape
for i in range(X.shape[1]):
    X_scaled[:, i] = scaler.fit_transform(X[:, i].reshape(-1, 1)).flatten()


# In[6]:


#Step-6: Reshape data for CNN-LSTM input
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))


# In[7]:


def create_sequences(data, labels, window_size=30):
    X_seq, y_seq = [], []
    for i in range(len(data) - window_size):
        X_seq.append(data[i:i+window_size])
        y_seq.append(labels[i+window_size])  
    return np.array(X_seq), np.array(y_seq)


# In[8]:


window_size = 30  
X_seq, y_seq = create_sequences(X_scaled, y, window_size)


# In[9]:


# Step-7:Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)


# In[10]:


# Step-8:Model building
model = Sequential()

# CNN Feature Extraction
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, X_seq.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# BiLSTM for Temporal Dependencies
model.add(Bidirectional(LSTM(50, return_sequences=True)))  
model.add(LSTM(25, return_sequences=False))  
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))  
model.add(Dense(1, activation='linear'))  
# now compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


# In[11]:


#Step-9:Early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)


# In[ ]:


#Step-10:Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, lr_scheduler])


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


# step-11: Compute RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")


# In[ ]:


#Step-12: Plot actual vs predicted TTD
plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label="Actual TTD", marker='o', linestyle='')
plt.plot(y_pred[:100], label="Predicted TTD", marker='x', linestyle='')
plt.title("Actual vs Predicted Time to Disruption (TTD)")
plt.xlabel("Sample Index")
plt.ylabel("TTD (ms)")
plt.legend()
plt.show()


# In[ ]:


# Step-13: Save the trained model
model.save('disruption_ttd_model.h5')


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label="Actual TTD", marker='o', linestyle='-', alpha=0.7)
plt.plot(y_pred[:100], label="Predicted TTD", marker='x', linestyle='-', alpha=0.7)
plt.title("Actual vs Predicted Time to Disruption (TTD)")
plt.xlabel("Sample Index")
plt.ylabel("TTD (ms)")
plt.legend(loc="upper right", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# In[ ]:


#Step13: plot the MAE AND LOss graphs
# Extract training history
loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(loss) + 1)

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Plot MAE (Mean Absolute Error)
plt.figure(figsize=(10, 5))
plt.plot(epochs, mae, 'bo-', label='Training MAE')
plt.plot(epochs, val_mae, 'ro-', label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Training and Validation MAE')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# In[ ]:


df[['Plasmact', 'H-alpha', 'Mirnov', 'DeltaR', 'SXR', 'TTD']].corr()


# In[ ]:


#plot the correlation and Confusion matrix
import seaborn as sns
plt.figure(figsize=(8,6))
sns.heatmap(df[['Plasmact', 'H-alpha', 'Mirnov', 'DeltaR', 'SXR', 'TTD']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# In[ ]:


X_test_perturbed = X_test.copy()
X_test_perturbed += np.random.normal(0, 0.01, X_test.shape)

y_pred_perturbed = model.predict(X_test_perturbed)
perturb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_perturbed))
print(f"RMSE with perturbed input: {perturb_rmse:.4f}")


# In[ ]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

threshold = 5.0

y_test_classes = (y_test <= threshold).astype(int)
y_pred_classes = (y_pred.flatten() <= threshold).astype(int)

cm = confusion_matrix(y_test_classes, y_pred_classes)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test_classes, y_pred_classes))
acc = accuracy_score(y_test_classes, y_pred_classes)
print(f"\nAccuracy: {acc:.4f}")
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disruption', 'Disruption'],
            yticklabels=['No Disruption', 'Disruption'])
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:


#Testing the model on unseen dataset


# In[ ]:


import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("disruption_ttd_model.h5")


unseen_data = pd.read_csv("C:\\Users\\mycha\\OneDrive\\Documents\\my final project\\allfiles33\\csv\\test\\final_aligned_testdataset_fixed15.csv")  


selected_columns = ["Plasmact","H-alpha", "Mirnov", "DeltaR", "SXR"]
unseen_features = unseen_data[selected_columns].values


X_unseen_scaled = np.zeros_like(unseen_features)  
for i in range(unseen_features.shape[1]):
    X_unseen_scaled[:, i] = (unseen_features[:, i] - unseen_features[:, i].min()) / \
                            (unseen_features[:, i].max() - unseen_features[:, i].min())

X_unseen_scaled = X_unseen_scaled.reshape((X_unseen_scaled.shape[0], X_unseen_scaled.shape[1], 1))


def create_sequences(data, window_size=30):
    X_seq = []
    for i in range(len(data) - window_size):
        X_seq.append(data[i:i+window_size])
    return np.array(X_seq)


window_size = 30  
X_unseen_seq = create_sequences(X_unseen_scaled, window_size)


y_pred_unseen = model.predict(X_unseen_seq)


time_values = unseen_data["Time"].values[window_size:]  

predicted_disruption_index = np.argmin(y_pred_unseen) 
predicted_disruption_time = time_values[predicted_disruption_index]

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




