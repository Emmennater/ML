
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

""" Columns
           Area - The number of pixels within the boundaries of the raisin (these are photos of raisins)
MajorAxisLength - Length of the longest axis in pixels
MinorAxisLength - Length of the shortest axis in pixels
   Eccentricity - Ranges from 0 to 1; 0 is a circle, 1 is highly elongated
     ConvexArea - The area in pixels of the smallest convex shape that encapsulates the raisin
         Extent - Ranges from 0 to 1; Ratio of Area to Bounding Box
      Perimeter - In pixels
          Class - Type of raisin (Kecimen/Besni)
"""

# https://www.kaggle.com/datasets/nimapourmoradi/raisin-binary-classification
df = pd.read_csv('Raisin_Dataset.csv')
df[df.select_dtypes("int64").columns] = df[df.select_dtypes("int64").columns].astype("float64")
df['Class'] = df['Class'].apply(lambda x: 0 if x == 'Kecimen' else 1)
X = df.drop('Class', axis=1)
y = df['Class']
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

kecimen_count = len(df[df['Class'] == 0])
besni_count = len(df[df['Class'] == 1])
print(f"Kecimen: {kecimen_count}")
print(f"Besni: {besni_count}")

default_gamma = 1.0 / len(X.columns)
print(f"Default Gamma: {default_gamma}")

params = {
    'C':[0.1, 0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.5, 5.0, 6.0, 8.0, 10.0],
    # 'C':np.linspace(5,7,11),
    'gamma':np.linspace(default_gamma/10,default_gamma*10, 10)
    # 'gamma':[0.1,0.2,0.3,0.4,default_gamma,0.6,0.7,0.8,0.9,1.0]
    # 'gamma':np.linspace(0.55,0.65,11)
}

model = SVC(kernel='rbf', decision_function_shape='ovo') # Radial basis function, One vs one

# Using accuracy as the scoring function may not always be the best way to fit the model
# it may find the best accuracy for the train set, but that doesn't guarantee the test accuracy will be better
# I am using accuracy because the classes are perfectly balanaced.
# If they were unbalanced I could have used balanced_accuracy (average recall)
# Remember that recall is the % correct of the true yes population
gs = GridSearchCV(model, param_grid=params, scoring='accuracy')
gs.fit(X_train, y_train)
print(f"Best params: {gs.best_params_}")
print(f"Accuracy (train): {gs.best_estimator_.score(X_train, y_train)}")
print(f"Accuracy (test): {gs.best_estimator_.score(X_test, y_test)}")
# model = SVC(C=gs.best_params_['C'],gamma=gs.best_params_['gamma'])
# model.fit(X_train, y_train)
# print(f"Accuracy (test): {model.score(X_test, y_test):.3f}")

cm = confusion_matrix(y_test, gs.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=['Kecimen', 'Besni'])
disp_cm.plot()
plt.show()
