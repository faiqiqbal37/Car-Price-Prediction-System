

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import KFold


from sklearn.model_selection import cross_val_score



df = pd.read_csv("CarPrice_Assignment.csv")
df.head()
df.describe()

df1 = df.drop(['symboling','enginelocation','wheelbase','curbweight','enginetype','enginelocation','fuelsystem','boreratio','stroke','compressionratio','peakrpm'], axis = 'columns')
df1.head()

df1.isnull().sum()
print(df1['carbody'].unique())

# Correct the car names and a visualization of amount of the cars

df1 = df1.drop(['car_ID'],axis=1)
df1['CarName'] = df1['CarName'].str.split(' ',expand=True)[0]
df1['CarName'] = [i.lower() for i in df1["CarName"]]
df1['CarName'] = df1['CarName'].replace({'maxda': 'mazda',
                                       'porcshce': 'porsche',
                                       'toyouta': 'toyota',
                                       'vokswagen': 'volkswagen',
                                       'vw': 'volkswagen'})

car_brands = pd.DataFrame(df1['CarName'].value_counts()).reset_index().rename(columns={'index':'car brand','CarName': 'amount'})
plt.figure(figsize=(8, 8))
sns.barplot(x="amount", y="car brand", data=car_brands)
plt.title("Amount of Cars")
plt.grid(axis="x")
plt.tight_layout()
plt.show()

# Encoding all the categorical alphabetical attributes to numeric attributes

obj_df = df1.select_dtypes(include=['object']).copy()
obj_df.head()

obj_df["CarName"] = obj_df["CarName"].str.replace(' ', '') 
obj_df["CarName"] = obj_df["CarName"].astype('category')
obj_df.dtypes
obj_df["CarName_Labels"] = obj_df["CarName"].cat.codes

obj_df["fueltype"] = obj_df["fueltype"].astype('category')
obj_df.dtypes
obj_df["fueltype_Labels"] = obj_df["fueltype"].cat.codes

obj_df["aspiration"] = obj_df["aspiration"].astype('category')
obj_df.dtypes
obj_df["aspiration_Labels"] = obj_df["aspiration"].cat.codes

obj_df["doornumber"] = obj_df["doornumber"].astype('category')
obj_df.dtypes
obj_df["doornumber_Labels"] = obj_df["doornumber"].cat.codes

obj_df["carbody"] = obj_df["carbody"].astype('category')
obj_df.dtypes
obj_df["carbody_Labels"] = obj_df["carbody"].cat.codes

obj_df["drivewheel"] = obj_df["drivewheel"].astype('category')
obj_df.dtypes
obj_df["drivewheel_Labels"] = obj_df["drivewheel"].cat.codes

obj_df["cylindernumber"] = obj_df["cylindernumber"].astype('category')
obj_df.dtypes
obj_df["cylindernumber_Labels"] = obj_df["cylindernumber"].cat.codes


obj_df.head(10)

# Dropping the alphabetical attributes and using the encoded attributes

df2 = obj_df.drop(['CarName','fueltype','aspiration','doornumber','carbody','aspiration','doornumber','carbody','drivewheel','cylindernumber'], axis = 'columns')
df2.head(200)

df3 = df1.drop(['CarName','fueltype','aspiration','doornumber','carbody','aspiration','doornumber','carbody','drivewheel','cylindernumber'], axis = 'columns')


df4 = pd.concat([df2,df3], axis = 'columns')
df4.head()

X = df4.drop('price', axis = 'columns')
X.head()

y = df4.price
y.head()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state = 10)

# Using Linear regression model

lr = LinearRegression()
lr.fit(X_train,y_train)
lrScore = lr.score(X_test,y_test)

print("The Accuracy Score Of Linear Regression Model Is: ",lrScore )

#Using Decsion Tree regression Model

regressor = DecisionTreeRegressor()
regressor.fit(X_train,y_train)
DTScore = regressor.score(X_test,y_test)
print("The Accuracy Score Of Decision Tree Regression Is: ",DTScore )

# Applying the k-fold cross validation

cv = KFold(n_splits = 3, shuffle = True)
lrKfoldScore = cross_val_score(LinearRegression(), X, y, cv = cv)
print("The accuracy score of linear regression after applying Kfold is: ", lrKfoldScore)

cv = KFold(n_splits = 3, shuffle = True)
DTKfoldScore = cross_val_score(DecisionTreeRegressor(), X, y, cv = cv)
print("The accuracy score of Decision Tree Regression after applying Kfold is: ", DTKfoldScore)


models = ['Linear Regression', 'Decision Tree Regressor']
score = [59,80]

plt.bar(models,score)
plt.title('Accuracy Of The Models')
plt.xlabel('Regression Models')
plt.ylabel('Scores')
plt.show()


# Choosing Decision Tree Regressor as final model
predicted = regressor.predict(X_test)



# So using Decision Tree Regression As It is more accurate than simple linear regression

pickle.dump(regressor, open('./model.sav', 'wb'))


print("\n\n The Predicted Values Are: \n", predicted)

# Original data of X_test

expected = y_test

# Regression coefficients

print('Original Values Are:\n',expected)


# variance score: 1 means perfect prediction

print('Variance score: ',regressor.score(X_test, y_test))




# Plot a graph for expected and predicted values

plt.title('ActualPrice Vs PredictedPrice (Used Car Price Prediction)')
plt.scatter(expected,predicted,c='b',marker='.',s=36)
plt.plot([0, 50], [0, 50], '--r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


