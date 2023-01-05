from statistics import mean
import pandas as pd
from imblearn.pipeline import Pipeline
from matplotlib import pyplot as plt, pyplot
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn import preprocessing, svm
from imblearn.over_sampling import RandomOverSampler

# Merge two docs about Toronto
crime_csv = pd.read_excel(r"C:\Users\rinat\OneDrive\Desktop\SCHOOL STUFF\Machine Learning\Toronto\crime_toronto.xlsx")
toronto_info = pd.read_excel(
    r"C:\Users\rinat\OneDrive\Desktop\SCHOOL STUFF\Machine Learning\Toronto\Toronto_NBH_data.xlsx")
merge_toronto = pd.merge(crime_csv, toronto_info, left_on='Neighbourhood', right_on='Neighbourhood', how='left')

# Drop Nieghbrouhood labels and fill na with integer value
merge_toronto = merge_toronto.drop(['Neighbourhood', 'Neighbourhood Number'], axis=1)
merge_toronto = merge_toronto.fillna(1000000000)
merge_toronto = merge_toronto.astype(int)


def whitespace_remover(dataframe):
    for i in dataframe.columns:
        if dataframe[i].dtype == 'object':
            dataframe[i] = dataframe[i].map(str.strip)
        else:
            pass


# applying whitespace_remover function on dataframe
whitespace_remover(merge_toronto)

# Drop columns and rows with all nan
merge_toronto.drop([col for col, val in merge_toronto.sum().items() if val == 1000000000 * 140], axis=1, inplace=True)
merge_toronto.drop(merge_toronto.index[19], axis=0, inplace=True)
merge_toronto.drop(merge_toronto.index[77], axis=0, inplace=True)
merge_toronto.drop(merge_toronto.index[89], axis=0, inplace=True)
merge_toronto.drop(merge_toronto.index[127], axis=0, inplace=True)

# Create excel of complete dataset
merge_toronto.to_excel('test.xlsx')

# Drop target column
homicide_df = pd.DataFrame(merge_toronto).drop('Homicide_2016', axis=1)
homicide_all_features = homicide_df.columns.values.tolist()

# Set X and Y
X = homicide_df
X_normalized = preprocessing.normalize(X, norm='l2')
homicide_target_cols = ['Homicide_2016']
y = (merge_toronto[homicide_target_cols].astype(int).values.ravel())

# Apply Oversampling
ros = RandomOverSampler(random_state=42)
x_ros, y_ros = ros.fit_resample(X_normalized, y)

'''
#Visualize Y class distribution before and after oversampling:

merge_toronto['Homicide_2016'].value_counts().plot.bar()
plt.title('Homicide 2016', fontsize=18)
plt.xlabel('Counts of Homicide', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()

target = pd.DataFrame(y_ros)
target.value_counts().plot.bar()
plt.title('Oversampled_Homicide 2016', fontsize=18)
plt.xlabel('Counts of Homicide', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.show()
'''

# Create Test/Train split
X_train, X_test, y_train, y_test = train_test_split(x_ros, y_ros, test_size=0.2,
                                                    random_state=1)  # 80% training and 20% test
# Apply svm classifier
clf = svm.SVC(decision_function_shape='ovo')

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Create RFE pipeline
rfe = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=5)
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s', rfe), ('m', model)])

# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=5)
n_scores = cross_val_score(pipeline, x_ros, y_ros, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print((mean(n_scores)))

# summarize all features
rfe.fit(X, y)
features = []
for i, j in zip(range(X.shape[1]), X.columns):
    if rfe.support_[i] == True:
        features.append(j)
        print(i, rfe.support_[i], rfe.ranking_[i])
print(features)
