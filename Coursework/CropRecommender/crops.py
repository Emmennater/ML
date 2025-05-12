import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

""" Data Fields
N - percent of Nitrogen content in soil
P - percent of Phosphorous content in soil
K - percent of Potassium content in soil
temperature - temperature in degree Celsius
humidity - relative humidity in %
ph - ph value of the soil
rainfall - rainfall in mm
"""

# https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
df = pd.read_csv('crops-cleaned.csv')

predict = 'label'
class_names = df['label'].unique().tolist()

# Select variables
y = df[predict]
X = df.drop(predict, axis=1)

# Train/test split with 40% left as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Better to use None because there is the same amount of data for each type of crop
clf = DecisionTreeClassifier() # lower max depth reduces overfitting
parameters = {"max_depth": range(2, 16), "class_weight": [None, "balanced"]}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)
max_depth = grid_search.best_params_["max_depth"]
class_weight = grid_search.best_params_["class_weight"]
clf = DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight)
clf.fit(X_train, y_train)

print(f"Best Params: {grid_search.best_params_}")
print(f"Train Score: {clf.score(X_train, y_train):.3f}")
print(f"Test Score: {clf.score(X_test, y_test):.3f}")

fig, ax = plt.subplots(figsize=(8,6))
tree.plot_tree(clf, fontsize=8, max_depth=3, feature_names=X.columns, class_names=class_names, filled=True)
fig.set_tight_layout(True)
plt.show()
