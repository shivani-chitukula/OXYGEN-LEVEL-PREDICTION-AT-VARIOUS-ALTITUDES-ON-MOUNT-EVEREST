# OXYGEN-LEVEL-PREDICTION-AT-VARIOUS-ALTITUDES-ON-MOUNT-EVEREST
The objective of this project is to develop a machine-learning model that can accurately predict oxygen levels at different altitudes on Mount Everest. 
To achieve this, I trained different machine learning models by taking various environment variables as input. 
The dataset is taken from the National Geographic website. The data is organized in the CSV files by date and contain hourly readings. There are six different CSV files in this data. 
PHORTSE (3,810 m), BASE CAMP (5,315 m), CAMP 2(6,464 m), SOUTH COL (7,945 m), BALCONY (8,430 m), and BISHOP ROCK (8,810 m). Each CSV file contains environmental variables such as temperature, pressure, and relative humidity. 
The data is available for each hour of the day within a specific time period. 
A single representative tuple each day (at 12:00 pm in 24 hours), 6 slots a day, 4 slots a day, 2 slots a day, and all tuples are taken for analysis. For each analysis task, a dataset is generated. Preprocessing is performed for each dataset and selected features based on the correlation matrix. Two columns are added, "oxypercent" and "height". Oxypercent is filled using a formula. 
These six datasets are concatenated. Missing data is handled using ffill. Features are selected based on correlation matrix.
Several machine learning algorithms were tested for different analysis tasks, including decision tree regressor, random forest regressor, Ridge regression, Lasso regression and linear regression. Models are evaluated by using metrics such as mean square error, mean absolute error, and r-squared error. Suitable model is selected.
In conclusion,for Single representative tuple: Decision tree regressor
 Six slots a day: Random forest regressor
Four slots a day: Decision tree regressor
Two slots a day: Linear regression
All tuples: Linear regression,are selected.
The ultimate goal of the project is to help improve the safety and success rate of mountaineering on Mount Everest.
