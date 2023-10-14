# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.
   
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SACHIN.C
RegisterNumber:  212222230125

```

```PYTHON
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### DATA.HEAD()
![image](https://github.com/Sachin-vlr/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497666/f03f6fb7-787d-45fe-b6c3-9371162ede24)

### DATA.INFO()
![image](https://github.com/Sachin-vlr/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497666/52b3cb76-4c83-46c5-890d-250e38f74d06)

### ISNULL() AND SUM()
![image](https://github.com/Sachin-vlr/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497666/c428ffdc-5525-46e7-8d71-14d447af335a)

### DATAVALUE COUUNTS()
![image](https://github.com/Sachin-vlr/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497666/01f94040-f0e1-4819-8e7b-17c04a81ee9e)

### Data.head() for salary
![image](https://github.com/Sachin-vlr/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497666/e8b5abc2-996a-4083-a32b-55f675b3be0c)

### X.Head()
![image](https://github.com/Sachin-vlr/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497666/9beb7401-923f-4d6a-ba94-beaa8603a319)

### Accuracy Value
![image](https://github.com/Sachin-vlr/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497666/fd2af46f-b2aa-4a2a-91a8-caff21b5fd0d)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
