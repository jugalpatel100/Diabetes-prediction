import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle

df = pd.read_csv('diabetes.csv')

columns_to_consider = ['Pregnancies', 'Glucose', 'BloodPressure','Insulin', 'BMI', 'Age']

x_train, x_test, y_train, y_test = train_test_split(df[columns_to_consider], df['Outcome'], random_state=0)

clf = GradientBoostingClassifier(learning_rate=0.2, n_estimators=65, min_samples_split=80)

clf.fit(x_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))

'''model = pickle.load(open('model.pkl', 'rb'))

p = model.predict_proba([[1, 115, 70, 96, 35, 32]])

print(p)'''