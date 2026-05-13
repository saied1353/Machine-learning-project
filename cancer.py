import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report)

dataset=pd.read_csv(r"D:\chrome down\archive (10).zip")
d=dataset.head(5)
print(d)
s=dataset.shape
print(s)
x=dataset.drop(dataset.columns[1],axis=1)
y=dataset.iloc[:,1]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(dataset.iloc[:,1])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
re=RandomForestClassifier( n_estimators=300,
    max_depth=10,
    min_samples_split=9,
    min_samples_leaf=9,
    random_state=42,
    class_weight='balanced')
re.fit(x_train,y_train)
y_pred =re.predict(x_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("Train Accuracy:", re.score(x_train, y_train) * 100)
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred, average="weighted"))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

