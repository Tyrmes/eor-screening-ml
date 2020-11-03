
# 1) Import Python Packages for Data Science

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px

# 2) Import the Dataset
data = pd.read_csv('DATA WORLWIDE EOR PROJECTSP.csv')

# 3) Data Preparation
# 3.1) Exploratory Data Analysis (EDA)
data.info() # Data description
data.describe() #Summary Statistics

# 3.2) Visualizations
# Boxplots of the categorical variable EOR method vs its feature variables
px.box(data, x="EOR_Method", y="Permiability")
px.box(data, x="EOR_Method", y="Depth")
px.box(data, x="EOR_Method", y="Gravity")
px.box(data, x="EOR_Method", y="Oil_Saturation")

# Scatter plots of some feature variables
px.scatter(data,  x="Temperature", y="Permiability")
px.scatter(data,  x="Temperature", y="Depth")
px.scatter(data,  x="Temperature", y="Oil_Saturation")
px.scatter(data,  x="Temperature", y="Gravity")

# Pair plot where is shown scatter plots of all feature variables, as well as the statistic distribution of each of them in the diagonal
sns.pairplot(data, hue="EOR_Method")

# 3.3) Data Preprocessing
# Heatmap to identify presence of null data
sns.heatmap(data.isnull())

# Matrix Correlation for checking the correlation among the feature variables
f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
               square=True, ax=ax)

# Feature Selection (Feature importance)
X = data.iloc[:,2:9].values #Feature Variables
y = data.iloc[:,1:2].values # Target Variable

# The next codes are used to desploy the importance between each feature variable with the target variable
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(7).plot(kind='barh')
plt.show()

# Data transforming
from sklearn.preprocessing import MinMaxScaler # Normalization of the numerical variables
sc = MinMaxScaler()
X = sc.fit_transform(X)
from sklearn.preprocessing import LabelEncoder # Data Labeling for categorical variable
le = LabelEncoder()
dfle = data
dfle.EOR_Method = le.fit_transform(dfle.EOR_Method)
from sklearn.preprocessing import OneHotEncoder # Encoding of categorical variable
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

# 4) Data Split and model training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

# 4.1) Model training using the KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors = 2, p=2, metric= 'euclidean')
Knn.fit(X_train, y_train)

# 4.2) Model training using the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
arbol = DecisionTreeClassifier(max_depth = 1)
arbol.fit(X_train, y_train)

# 5) Model Evaluation
# 5.1) KNN Algorithm
Knn.score(X_train, y_train)
Knn.score(X_test, y_test)

# Plot of Accuracy vs K values using the training and testing data
neighbors = np.arange(1, 30)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)
plt.plot(neighbors, test_accuracy, label = 'Test')
plt.plot(neighbors, train_accuracy, label = 'Training')
plt.legend()
plt.xlabel('K_neighbors')
plt.ylabel('Accuracy')
plt.show()

# 5.2) DT Algorithm
arbol.score(X_train, y_train)
arbol.score(X_test, y_test)

# Plot of Accuracy vs max depth values using the training and testing data
max_depth = np.arange(1, 9)
train_accuracy = np.empty(len(max_depth))
test_accuracy = np.empty(len(max_depth))
for i, r in enumerate(max_depth):
    Tree = DecisionTreeClassifier(max_depth=r)
    Tree.fit(X_train, y_train)
    train_accuracy[i] = Tree.score(X_train, y_train)
    test_accuracy[i] = Tree.score(X_test, y_test)
plt.plot(max_depth, test_accuracy, label = 'Test')
plt.plot(max_depth, train_accuracy, label = 'Train')
plt.legend()
plt.xlabel('max_depth_values')
plt.ylabel('Accuracy')
plt.show()

# 6) Model deployment (Prediction)
def predict(Porosity=29, Permiability=4689, Depth=1200 ,Gravity=8, Viscocity=490058, Temperature=150, Oil_Saturation=38.4):
  cnames = ['Porosity', 'Permiability', 'Depth', 'Gravity', 'Viscocity', 'Temperature', 'Oil_Saturation']
  data = [[Porosity, Permiability, Depth, Gravity, Viscocity, Temperature, Oil_Saturation]]
  my_X = pd.DataFrame(data=data, columns=cnames)
  my_X = sc.transform(my_X)
  return Knn.predict(my_X)

if np.argmax(predict())==0:
    print('CO2 Injection')
elif np.argmax(predict())==1:
    print('Combustion')
elif np.argmax(predict())==2:
    print('HC Injection')
elif np.argmax(predict())==3:
    print('Polymer')
elif np.argmax(predict())==4:
    print('Steam Injection')




