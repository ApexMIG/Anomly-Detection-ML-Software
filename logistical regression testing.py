import pandas as pd
from sklearn import linear_model
import numpy as np

#Reading the CSV files
data_wrong_tot = pd.read_csv("location-data-anomaly.csv")
data_right_tot = pd.read_csv("location-data.csv")

#randomly selecting index numbers to split the dat into two data sets of 80/20 split of test and training data
rand_right = np.random.randint(0,100,20)
rand_wrong = np.random.randint(0,100,20)

data_right_test = data_right_tot.iloc[rand_right]
data_wrong_test = data_wrong_tot.iloc[rand_wrong]
data_right_train = data_right_tot.drop(rand_right)
data_wrong_train = data_wrong_tot.drop(rand_wrong)

#Combining the right and wrong datasets to make a training dataset and a testing dataset
frames = [data_right_train,data_wrong_train]
data_train = pd.concat(frames)
data_train['anomaly'] = data_train['anomaly'].astype(int)

fraame = [data_right_test,data_wrong_test]
data_test = pd.concat(fraame)
data_test['anomaly'] = data_test['anomaly'].astype(int)

#Using the Training Dataset to create a Machine Learning algorithm Based on Logical Regression
features = ["latitude","longitude"]
X =data_train[features]
y =data_train["anomaly"]

reg = linear_model.LogisticRegression()
reg.fit(X.values,y)

#Testing to see the accuracy of the Machine Learing Algorithm by using the set aside test data
correct =  0
total = len(data_test)
for index,row in data_test.iterrows():
    x = reg.predict([[row["latitude"],row["longitude"]]])
    if x == row["anomaly"]:
        correct = correct + 1
print(f"Out of {total} answers only {correct} were right")
Accuracy_percentage = correct/total*100
print("It has an accurary rate of",Accuracy_percentage)

# original percentage was 57.9 but ever since random are taken it dropped to in between 45.5 and 57.5