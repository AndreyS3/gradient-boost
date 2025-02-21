#!/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42

bunch = load_breast_cancer(as_frame=True)
X = bunch.data
y = bunch.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

forest = RandomForestRegressor(n_estimators=20, random_state=RANDOM_STATE)
forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
#print(f"y_pred: {y_pred}")

thresholds = np.linspace(0., 1.1, 21)

tpr = np.zeros(len(thresholds))
fpr = np.zeros(len(thresholds))
y_test_bool = (y_test == 1).values

for idx, th in enumerate(thresholds):
    y_class = (y_pred >= th)
    tp = (y_class & y_test_bool).sum()
    tn = (~y_class & ~y_test_bool).sum()
    fp = (y_class & ~y_test_bool).sum()
    fn = (~y_class & y_test_bool).sum()
    tpr[idx] = tp / (tp + fn)
    fpr[idx] = fp / (fp + tn)

    #print(f"th: {th}, y_class: {y_class}")
    #print(f"{th}: tp={tp}, tn={tn}, fp={fp}, fn={fn}")

#print(tpr)
#print(fpr)

plt.plot(fpr, tpr, 'o-', linewidth=2)
plt.show()

