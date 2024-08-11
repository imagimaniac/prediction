

# %%
# Importing relevant important Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %%
df = pd.read_csv('arPrice_Assignment.csv')
df
#%%
def categories():
    for i in range(len(df.columns)):
        #if df[df.columns[i]].nunique() < 20:
        if df[df.columns[i]].nunique():
            if df[df.columns[i]].nunique() <= 7: 
                print("Number of Unique categories in",df.columns[i],df[df.columns[i]].nunique())
categories()

#%%
dummies_df = pd.get_dummies(df[["symboling","fueltype","aspiration","doornumber","carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber"]])
dummies_df

#%%
final_df = pd.concat([df, dummies_df],axis=1)

final_df = final_df.drop(["CarName","symboling","fueltype","aspiration","doornumber","carbody", "drivewheel", "enginelocation", "enginetype", "cylindernumber","fuelsystem"],axis=1)


#%%

X = final_df.drop(columns=["price"], axis=1)
y = final_df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#%%

#Import Required Libraries for Linear Regression

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.linear_model import LinearRegression


mlin_regr = linear_model.LinearRegression()
mlin_regr.fit(X_train, y_train)

# Save the model to a pickle file
with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(mlin_regr, file)



#%%