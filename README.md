The Nuclear fusion reactor face a significant 
risk due to the disruption which results in loss of the 
plasma confinement and severe damage to walls of the 
fusion reactor. Therefore predicting such disruptions 
in early stage is very essential to prevent the damage.
This research work presents a machine learning model 
for predicting the disruptions in the early which is 
developed using the non-stationary signals, including
H-alpha, Mirnov coil, Soft X-ray(SXR), DeltaR signals.
Most of the traditional based approaches rely on the 
Plasma current for the disruption prediction, this 
model mainly focuses on the variations in the non-
sationary signals to predict the disruptions even before 
they occur. The proposed model work on the 
Convolutional Neural networks(CNN) to extract the 
spatial features from the input signals and a 
Bidirectional Long Short-Term Memory (BiLSTM) 
network to capture the time dependencies in the 
signnals.The machine learning model is trained on 
experimental data and for evaluating the model the 
performance metrics like RMSE(root mean square 
error) and R² score were used, ensuring high accuracy 
and robustness .Min-Max scaling and feature 
engineering techniques were used for data 
preprocessing. This research work provides a potential 
,reliable, real-time disruption prediction system for 
preventive measures to mitigate potential damage.
