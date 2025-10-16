#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the dataset (Modify the file path if needed)
df = pd.read_csv("C:\\Users\\mycha\\OneDrive\\Documents\\my final project\\allfiles33\\final_aligned_dataset_fixed15.csv")  # Uncomment if reading from a file

# Ensure 'Time' column is float
df['Time'] = df['Time'].astype(float)

# Get the first occurrence of disruption (where Label == 1)
first_disruption_time = round(df[df['Label'] == 1]['Time'].min(), 5)
print("First Disruption Time:", first_disruption_time)

# Compute Time to Disruption (TTD)
df['TTD'] = (first_disruption_time - df['Time']).clip(lower=0)

# Display some sample output
print(df[['Time', 'TTD']].head(10))

# Save the modified dataset (optional)
df.to_csv("C:\\Users\\mycha\\OneDrive\\Documents\\my final project\\allfiles33\\csv\\ttdcheck15.csv", index=False)  # Uncomment to save


# In[6]:


df.head()


# In[ ]:





# In[ ]:




