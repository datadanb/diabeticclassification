
# coding: utf-8

# In[1]:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

#Read in data and provide some visualisations for each feature/variable
df = pd.read_csv("pima_diabetes.csv")

#Describe the data out
df.describe()

b = df.astype(bool).sum(axis=0)

b.plot(kind='bar')
plt.show()




# In[3]:
d = df.groupby('Outcome').size()

d.plot.pie(figsize = (6,6),title ='Outcome Class Breakdown', fontsize = 20)

plt.show()


# In[2]:
#Show shape/data for columns of interest

pd.DataFrame.hist(data=df, column= 'Age')
pd.DataFrame.hist(data=df, column= 'BloodPressure')
pd.DataFrame.hist(data=df, column= 'BMI')
pd.DataFrame.hist(data=df, column= 'DiabetesPedigreeFunction')
pd.DataFrame.hist(data=df, column= 'Glucose')
pd.DataFrame.hist(data=df, column= 'Insulin')

plt.show()


# In[23]:
#Dealing with missing data

df.head(20)

#Set BP to NaN where 0
dfclean1 = df.replace({'BloodPressure': {0: np.nan}})

#Impute with what is the previous value
dfclean1.fillna(method = 'bfill', inplace=True)

#Set BMI to NaN where 0
dfclean2 = dfclean1.replace({'BMI': {0: np.nan}})

#Impute with what is the previous value
dfclean2.fillna(dfclean2[['BMI']].mean(), inplace=True)

#Set GTT to NaN where 0
dfclean = dfclean2.replace({'Glucose': {0: np.nan}})

#Impute with what is the previous value
dfclean.fillna(dfclean[['Glucose']].mean(), inplace=True)

#Remove columns of no interest
del dfclean['SkinThickness']
del dfclean['Insulin']


dfclean.describe()



# In[6]:
#Display columns of interest (figures )

pd.DataFrame.hist(data=dfclean, column= 'BloodPressure')
pd.DataFrame.hist(data=dfclean, column= 'BMI')
pd.DataFrame.hist(data=dfclean, column= 'Glucose')
pd.DataFrame.hist(data=dfclean, column= 'DiabetesPedigreeFunction')
pd.DataFrame.hist(data=dfclean, column= 'Age')
pd.DataFrame.hist(data=dfclean, column= 'Pregnancies')

plt.show()


# In[24]:
#Show the format of each column
for column in dfclean.columns:
    print('{0} {1}'.format(str(type(dfclean[column][0])),str(column)))



# In[26]:
#Correlation matrix - As per figure 1 in Project document
import seaborn as sns

corrmat = dfclean.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.75, square=True)

print(corrmat)

plt.show()



# In[27]:
# Linear regression looking at Outcome & glucose, not used in the project stage 1
import statsmodels.api as sm

y = dfclean['Outcome']
X = dfclean[['Glucose','BloodPressure','BMI','Pregnancies','Age','DiabetesPedigreeFunction']]
X = sm.add_constant(X)
model11 = sm.OLS(y, X).fit()
model11.summary()


# In[11]:
import csv

#Writing file back out
outfile = dfclean.to_csv('C:\\Users\\dan bridgman\\Documents\\Uni\\Projects\\clean_diabetesdata.csv',mode = 'w+', sep = ',', index= False)
