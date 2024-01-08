import warnings
import pandas as pd

warnings.filterwarnings("ignore")
df = pd.read_csv("Datasets\Phortse_20230630.csv")

# convert TIMESTAMP column to datetime format
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

# set start and end date
start_date = '25-04-2019  12:00'
end_date = '31-07-2022  12:00'

# filter the dataframe based on the timestamp condition and hour
df_filtered = df.loc[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date) & (df['TIMESTAMP'].dt.hour == 12)]

# create a new subset with added columns height and oxypercent
subset1 = df_filtered
subset1['height']=3810
subset1['oxypercent']=None

# save the subset to a new csv file named final1.csv
print("success")
subset1.to_csv('Intermediate datasets\\final1.csv',index=False)



df = pd.read_csv("Datasets\Base_Camp_20230630.csv")
# convert TIMESTAMP column to datetime format
df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
# set start and end date
start_date = '19-10-2019  12:00:00'
end_date = '31-07-2022  12:00:00'
# filter the dataframe based on the timestamp condition and hour

df_filtered = df.loc[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date) & (df['TIMESTAMP'].dt.hour == 12)]

subset2 = df_filtered
# create a new subset with added columns height and oxypercent

subset2['height']=5315
subset2['oxypercent']=None
# save the subset to a new csv file named final2.csv
subset2.to_csv('Intermediate datasets\\final2.csv',index=False)



df = pd.read_csv("Datasets\Camp2_20230630.csv")
# convert TIMESTAMP column to datetime format

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
# set start and end date
start_date = '11-05-2019  00:00:00'
end_date = '31-07-2022  23:00:00'
# filter the dataframe based on the timestamp condition and hour

df_filtered = df.loc[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date) & (df['TIMESTAMP'].dt.hour == 12)]
subset3 = df_filtered
# create a new subset with added columns height and oxypercent

subset3['height']=6464
subset3['oxypercent']=None
# save the subset to a new csv file named final3.csv
subset3.to_csv('Intermediate datasets\\final3.csv',index=False)


df = pd.read_csv("Datasets\South_Col_20230630.csv")
# convert TIMESTAMP column to datetime format

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
# set start and end date
start_date = '23-05-2019  12:00:00'
end_date = '31-07-2022  12:00:00'
# filter the dataframe based on the timestamp condition and hour

df_filtered = df.loc[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date) & (df['TIMESTAMP'].dt.hour == 12)]
subset4 = df_filtered
# create a new subset with added columns height and oxypercent

subset4['height']=7945
subset4['oxypercent']=None
# save the subset to a new csv file named final4.csv

subset4.to_csv('Intermediate datasets\\final4.csv',index=False)


df = pd.read_csv("Datasets\Balcony_20210630.csv")
# convert TIMESTAMP column to datetime format

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
# set start and end date
start_date = '23-05-2019  12:00:00'
end_date = '09-02-2020  12:00:00'
# filter the dataframe based on the timestamp condition and hour

df_filtered = df.loc[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date) & (df['TIMESTAMP'].dt.hour == 12)]
subset5 = df_filtered
# create a new subset with added columns height and oxypercent

subset5['height']=8430
subset5['oxypercent']=None
# save the subset to a new csv file named final5.csv

subset5.to_csv('Intermediate datasets\\final5.csv',index=False)


df = pd.read_csv("Datasets\Bishop_Rock_20230630.csv")
# convert TIMESTAMP column to datetime format

df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
# set start and end date
start_date = '09-05-2022  12:00:00'
end_date = '09-07-2022  12:00:00'
# filter the dataframe based on the timestamp condition and hour

df_filtered = df.loc[(df['TIMESTAMP'] >= start_date) & (df['TIMESTAMP'] <= end_date) & (df['TIMESTAMP'].dt.hour == 12)]
subset6 = df_filtered
# create a new subset with added columns height and oxypercent

subset6['height']=8810
subset6['oxypercent']=None
# save the subset to a new csv file named final6.csv

subset6.to_csv('Intermediate datasets\\final6.csv',index=False)

df1 = pd.read_csv('Intermediate datasets\\final1.csv')
df2 = pd.read_csv('Intermediate datasets\\final2.csv')
df3 = pd.read_csv('Intermediate datasets\\final3.csv')
df4 = pd.read_csv('Intermediate datasets\\final4.csv')
df5 = pd.read_csv('Intermediate datasets\\final5.csv')
df6 = pd.read_csv('Intermediate datasets\\final6.csv')
#Then, we concatenate these dataframes vertically into a single dataframe using the pandas concat function.
df_concatenated = pd.concat([df1, df2,df3,df4,df5,df6])
#Finally, we save the concatenated dataframe as a CSV file named 'Mt_Everest.csv' without the index column.
df_concatenated.to_csv('Intermediate datasets\\Mt_Everest.csv',index=False)


import numpy as np
import math
df=pd.read_csv('Intermediate datasets\\Mt_Everest.csv')

# MISSING VALUES
df = df.replace(-999, np.nan)  # Replace the -999 values with NaN
df = df.replace(0.0, np.nan)   # Replace the 0 values with NaN
# df.fillna(df.fillna(df.mean()), inplace=True)  # Fill the NaN values with mean values of the column

df = df[['T_HMP', 'WDIR','WS_MAX','SW_IN_AVG','SW_OUT_AVG','LW_IN_AVG','LW_OUT_AVG','PRESS',  'height','RH']]
df.fillna(df.mean(), inplace=True)
df['oxypercent'] = float('nan')
# print(df.isnull().sum(),"HHHHHHHH")
# Function to calculate oxygen percentage
def calc_oxypercent(T_HMP, PRESS, RH, height):
    e_s = 6.112 * math.exp((17.67 * T_HMP) / (T_HMP + 243.5))  # Calculate the vapor pressure of water
    e = e_s * (RH / 100)
    P_d = PRESS - e
    T_K = T_HMP + 273.15
    R_d = 287.05
    g_0 = 9.80665
    M_air = 0.0289644
    L = -0.0065
    T_0 = T_K / ((P_d / 1013.25) ** (-R_d * L / g_0 / M_air))
    rho = P_d / (R_d * T_0)
    P_tot = PRESS / (math.exp(-height / 8200))
    P_o2 = 0.2095 * P_tot  # Update to use 20.95% as the oxygen concentration
    x_o2 = P_o2 / PRESS
    return 85-x_o2 *100

# Apply the above function to each row of the dataframe where 'oxypercent' is NaN, else use the existing value
df['oxypercent'] = df.apply(lambda x: calc_oxypercent(x['T_HMP'], x['PRESS'], x['RH'], x['height']) if math.isnan(x['oxypercent']) else x['oxypercent'], axis=1)

df.to_csv('Intermediate datasets\\analysis1.csv')  # Save the updated dataframe to a csv file


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
import joblib
# # Load the dataset
df = pd.read_csv("Intermediate datasets\\analysis1.csv")
# print(df.isnull().sum(),"HHHHHHHHHH")
# Split the dataset into training and testing sets

X = df[['T_HMP', 'WDIR','WS_MAX','PRESS',  'height']]
y = df['oxypercent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# df.dropna(X, inplace=True)

# Scale the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Create a model

model = DecisionTreeRegressor(max_depth=3)

# Fit the model to the training data
model.fit(X_train, y_train)
joblib.dump(model, 't_model.joblib')

# Make predictions on the test data
# y_pred = model.predict(X_test)

# new_data = pd.DataFrame({'T_HMP': [25],'WDIR':[129.8],'WS_MAX':[4.77],'SW_IN_AVG':[37.5],'SW_OUT_AVG':[19.35],'LW_IN_AVG':[332.15],'LW_OUT_AVG':[376.47], 'PRESS': [1013], 'height': 4000})
# new_data_scaled = scaler.transform(new_data)
# new_oxypercent = model.predict(new_data_scaled)
# print("Predicted oxygen percent:", new_oxypercent[0])


print("succeedd")
