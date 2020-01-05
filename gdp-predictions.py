from sklearn import *
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import model_selection
#from sklearn import grid_search


data = pd.read_csv("Kaggle.csv")


y = data["Gross domestic product GDP percapta"]
data = data[["Physicians per 10k people","Adolescent birth rate 15-19 per 100k 20102015","Mobile phone subscriptions per 100 people 2014",
"Homicide rate per 100k people 2008-2012","Birth registration funder age 5 2005-2013",
"Under-five Mortality 2013 thousands","Primary school dropout rate 2008-2014","Internet users percentage of population 2014",
"Prison population per 100k people","International inbound tourists thausands 2013",
"Domestic food price level 2009 2014 index","Female Suicide Rate 100k people","Primary school dropout rate 2008-2014",
"Taxes on income profit and capital gain 205 2013","General government final consumption expenditure - Perce of GDP 2005-2013",
"Under-five Mortality 2013 thousands","Research and development expenditure  2005-2012",
"Stock of immigrants percentage of population 2013","Exports and imports percentage GPD 2013","Consumer price index 2013","Secondary 2008-2014"]]


X_train, X_test, y_train, y_test = model_selection.train_test_split(data, y, test_size=0.15, random_state=42)


clf            = ExtraTreesRegressor()
parameters     = {'max_depth':np.arange(1,15)}
clfgrid        = model_selection.GridSearchCV(clf, parameters)
clfgrid.fit(X_train, y_train)
print(clfgrid.score(X_train,y_train))
print(clfgrid.score(X_test,y_test))

xx = clfgrid.predict(X_test)[:,np.newaxis]
y = y_test[:,np.newaxis]
print(np.concatenate((xx,y),axis=1))

