from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import numpy as np 
from perceptron import Perceptron
# load dataset
dataset = load_iris()

# slpit dataset
X = dataset.data[:, (2, 3)]
y = (dataset.target == 0).astype(np.int)

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)


# train 
perceptron = Perceptron(learning_rate=0.001, epochs=75)
perceptron.fit(X_train, y_train)

# predict
pred = perceptron.predict(x_test)

# accuracy 
print(accuracy_score(y_test, pred)) 
"""output:
1.0
"""

# report 
report = classification_report(y_test, pred, digits=2)
print(report)

"""output:
                precision    recall  f1-score   support

           0       1.00      1.00      1.00        48
           1       1.00      1.00      1.00        27

    accuracy                           1.00        75
   macro avg       1.00      1.00      1.00        75
weighted avg       1.00      1.00      1.00        75

"""