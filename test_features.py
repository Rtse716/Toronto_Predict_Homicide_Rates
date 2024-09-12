import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from mlxtend.evaluate import bias_variance_decomp

# Read the test.xlsx generated from the feature selection step. Replace with own file path.

newDataframe = pd.read_excel(r"C:\Users\rinat\OneDrive\Desktop\SCHOOL STUFF\Machine Learning\Toronto\test.xlsx")

# features from select_features.py
feature_cols = ['Assault_2016', '  Single-detached house', '      EI - Other benefits: Average amount ($)',
                '      Social assistance benefits: Average amount ($)',
                "      Social assistance benefits: Aggregate amount ($'000)"]

# Set X and Y
X = (newDataframe[feature_cols].astype(int))
X_normalized = preprocessing.normalize(X, norm='l2')
target_cols = ['Homicide_2016']
y = (newDataframe[target_cols].astype(int).values.ravel())

# Apply Oversampling
ros = RandomOverSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(X_normalized, y)

# Set training and testing split
X_train, X_test, y_train, y_test = train_test_split(x_ros, y_ros, test_size=0.26, random_state=1)

# kNN
nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=10)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(x_ros, y_ros)

'''
Other models to try:
clf = svm.SVC(decision_function_shape='ovr')
gp = GaussianProcessClassifier()
gnb = GaussianNB()
mn = MultinomialNB()
cb = ComplementNB()
'''

# Predict the response for test dataset
y_pred = nca_pipe.predict(X_test)

print(y_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))

# estimate bias and variance
avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(nca_pipe, X_train, y_train, X_test, y_test, loss='mse',
                                                            num_rounds=50, random_seed=20)

# summary of the results
print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)

# Confusion Matrix
cm = metrics.confusion_matrix(y_test, y_pred, normalize='all', labels=[0, 1, 2, 3, 4, 5, 6])
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.title('Confusion Matrix for Homicide_2016 Predictions')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
