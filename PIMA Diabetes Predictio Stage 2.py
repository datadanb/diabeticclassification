
# coding: utf-8

# In[30]:


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




# In[31]:

#Dealing with missing data

#df.head(20)

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



# In[32]:

# Linear regression looking at Outcome & glucose, not used in the project
import statsmodels.api as sm

y = dfclean['Outcome']
X = dfclean[['Glucose','BloodPressure','BMI','Pregnancies','Age','DiabetesPedigreeFunction']]
X = sm.add_constant(X)
model11 = sm.OLS(y, X).fit()
model11.summary()


# In[45]:

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


X = dfclean.iloc[:,0:6]
Y = dfclean.iloc[:,6]
select_top_4 = SelectKBest(score_func=chi2, k = 5)

fit = select_top_4.fit(X,Y)
features = fit.transform(X)

features[0:5]


# In[34]:

dfclean.head()


# In[46]:

from sklearn.preprocessing import StandardScaler

X_features = pd.DataFrame(data = features, columns = ["Glucose","Age","BMI","BloodPressure","Pregnancies"])

rescaledX = StandardScaler().fit_transform(X_features)

X = pd.DataFrame(data = rescaledX, columns= X_features.columns)

X.head()



# In[36]:

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# In[37]:

from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

X_train,X_test,Y_train,Y_test = train_test_split(X_features,Y, random_state = 10, test_size = 0.2)

scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)

# Set the parameters by cross-validation
tuned_parameters = [{'max_iter': [100,1000,10000],
                     'C': [1, 10, 100, 1000],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(LogisticRegression(C=1,max_iter = 1000), tuned_parameters, cv=10,
                       scoring='%s_macro' % score)
    clf.fit(X_train_transformed, Y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    


# In[47]:

# Declare models, split data & scale it
models = []

models.append(('Linear regression - newton', LogisticRegression(solver = 'newton-cg', C = 1, max_iter = 100)))
models.append(('Linear regression - liblinear',LogisticRegression(solver = 'liblinear', C = 1, max_iter = 100)))
models.append(('Support Vector Classification - Benchmark',SVC()))


# In[48]:

# Validation steps
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

X_train,X_test,Y_train,Y_test = train_test_split(X_features,Y, random_state = 1, test_size = 0.2)


scaler = StandardScaler().fit(X_train)

X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)


results = []
names = []

for name,model in models:
    kfold = KFold(n_splits=10, random_state=1)
    cv_result = cross_val_score(model,X_train_transformed,Y_train, cv = kfold,scoring = "accuracy")
    names.append(name + ' , Training')
    results.append(cv_result)
    cv_result = cross_val_score(model,X_test_transformed,Y_test, cv = 10,scoring = "accuracy")
    names.append(name + ' , Testing')
    results.append(cv_result)
for i in range(len(names)):      
    print('Mean cross validation results of + %s +  accuracy %0.22f (+/- %0.2f)' % (names[i],results[i].mean(),results[i].std()*2))
    

#cv_result = cross_val_score(LogisticRegression(solver = 'newton-cg', C = 1, max_iter = 100),X_test_transformed,Y_test, cv = 10,scoring = "accuracy")
#predicted = cross_val_predict(svc, X_test_transformed, Y_test, cv = 10)  
#cv_result = cross_val_score(svc,X_test_transformed,Y_test, cv = 10,scoring = "accuracy")


lr = LogisticRegression(solver = 'newton-cg', C = 1, max_iter = 100).fit(X_test_transformed, Y_test)
#svc = SVC().fit(X_train_transformed, Y_train)
#predicted = svc.predict(X_test_transformed)  
predicted = lr.predict(X_test_transformed)  


clasreport = classification_report(Y_test,predicted)

print(clasreport)
conf = confusion_matrix(Y_test,predicted)
label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)
print('%s - confusion matrix - Test' % names)
print(conf)
plt.show()


# In[50]:

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



title = "Logistic Regression"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=1)

#estimator = LogisticRegression(solver = 'newton-cg', C = 1, max_iter = 100)
estimator = SVC()

plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=10, n_jobs=4)

# title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = SVC(gamma=0.001)
# plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()

