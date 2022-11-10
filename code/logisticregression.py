# Importing the libraries

import numpy as np
import pandas as pd
import seaborn as sns

# Importing the datasets

df = pd.read_csv('../data/User_Data.csv')
df.isnull().sum()
print("--------------------USER DATA--------------------------")
print(df)
print("----------------------------------------------")

X = df.iloc[:, [2,3]].values
Y = df.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
#print(X_Train);
#print(X_Test);
#print(Y_Train);
#print(Y_Test);

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the Logistic Regression into the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_Train, Y_Train)

# Predicting the test set results

Y_Pred = classifier.predict(X_Test)
print("Y prediction")
print(Y_Pred)
print("----------------------------------------------")
# Making the Confusion Matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)
print("Confusion Matrix ");
print(cm)
print("----------------------------------------------")
sns.heatmap(pd.DataFrame(cm), annot=True)

from sklearn.metrics import accuracy_score
accuracy =accuracy_score(Y_Test, Y_Pred)

print("Accuracy score", accuracy)
print("----------------------------------------------")

