import streamlit as st 
import pandas as pd
from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

import joblib
model = joblib.load('t_model.joblib')
# new_data = pd.DataFrame({'T_HMP': [25],'WDIR':[129.8],'WS_MAX':[4.77],'SW_IN_AVG':[37.5],'SW_OUT_AVG':[19.35],'LW_IN_AVG':[332.15],'LW_OUT_AVG':[376.47], 'PRESS': [1013], 'height': 4000})
# new_data_scaled = scaler.transform(new_data)
# new_oxypercent = model.predict(new_data)
# print("Predicted oxygen percent:", new_oxypercent[0])
st.title("OXYGEN PREDICTION")
st.header("Enter Input Values:")
T_HMP = st.number_input("Temperature")
WDIR = st.number_input("Wind direction")
WS_MAX = st.number_input("Wind spped(max)")
PRESS = st.number_input("Pressure")
height = st.number_input("height")

if st.button("Predict Oxygen Percentage"):
    new_data = pd.DataFrame({'T_HMP': [T_HMP],
                        'WDIR': [WDIR],
                        'WS_MAX': [WS_MAX],
                        'PRESS': [PRESS],
                        'height': [height]})
    # st.write(new_data)

#     # Apply the same data preprocessing steps used during training (e.g., scaling)
#     scaler = joblib.load('t_model.joblib')
#     new_data_scaled = scaler.transform(new_data)

#     # Make prediction
    new_oxypercent = model.predict(new_data)
    st.success(f"Predicted Oxygen Percentage: {new_oxypercent[0]:.2f}%")