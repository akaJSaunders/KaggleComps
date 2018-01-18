
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

# Part 1 - Preprocessing of data

# Importing the dataset
train_set = pd.read_csv('datasets/train.csv')
test_set  = pd.read_csv('datasets/test.csv')

#Visualizing data
#sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train_set);
#sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train_set,
#              palette={"male": "blue", "female": "pink"},
#              markers=["*", "o"], linestyles=["-", "--"]);
              
#Cleaning data
def simplify_ages(data):
    data.Age = data.Age.fillna(-0.5)
    bins = (-1,0,5,12,18,25,35,60,120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(data.Age, bins, labels = group_names)
    data.Age = categories
    return data

def simplify_fares(data):
    data.Fare = data.Fare.fillna(-0.5)
    bins = (-1,0,8,15,31,1000)
    group_names = ['Unknown', 'Low', 'Middle-Low', 'Middle-High', 'High']
    categories = pd.cut(data.Fare, bins, labels = group_names)
    data.Fare = categories
    return data

def simplify_cabins(data):
    data.Cabin = data.Cabin.fillna('N')
    data.Cabin = data.Cabin.apply(lambda x: x[0])
    return data

def format_name(data):
    data['Lname']      = data.Name.apply(lambda x: x.split(' ')[0])
    data['NamePrefix'] = data.Name.apply(lambda x: x.split(' ')[1])
    return data

def family_size(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    return data

def is_alone(data):
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
    return data
    
def simplyfy_embarked(data):
    data['Embarked'] = data['Embarked'].fillna('S')
    return data

def simplyfy_prefix(data):
    data['NamePrefix'] = data['NamePrefix'].replace(['Lady.','Countness.','Capt.','Col.','Don.', 'Dr.', 'Major.'
                                                     'Rev.','Sir.','Jonkheer.','Dona.'], 'Rare')
    data['NamePrefix'] = data['NamePrefix'].replace('Mlle.','Miss.')
    data['NamePrefix'] = data['NamePrefix'].replace('Ms.','Miss.')
    data['NamePrefix'] = data['NamePrefix'].replace('Mme.','Mrs.')
    return data

def drop_features(data):
    return data.drop(['FamilySize','Ticket','Name','SibSp','Parch'], 1)

def transform_features(data):
    data = simplify_ages(data)
    data = simplify_fares(data)
    data = simplify_cabins(data)
    data = format_name(data)
    data = family_size(data)
    data = is_alone(data)
    data = simplyfy_embarked(data)
    data = simplyfy_prefix(data)
    data = drop_features(data)
    return data

train_set = transform_features(train_set)
test_set  = transform_features(test_set)


#Encoding features
from sklearn import preprocessing
def encode_features(train_set, test_set):
    features = ['Sex','Age','Fare','Cabin','Embarked','Lname','NamePrefix']
    data = pd.concat([train_set[features], test_set[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data[feature])
        train_set[feature] = le.transform(train_set[feature])
        test_set[feature]  = le.transform(test_set[feature])
    return train_set, test_set

train_set, test_set = encode_features(train_set, test_set)

#Separate the X and Y
y = train_set['Survived']
X = train_set.drop(['Survived','PassengerId'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

classifier = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(classifier, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
classifier = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
y_pred = classifier.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
#y_pred = classifier.predict(X_test)

#Predictions
predictions = classifier.predict(X_test)
print(accuracy_score(y_test, predictions))


from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold

print(cross_val_score(classifier, X,y,cv=KFold(891, n_folds=10)))

#Predict actual test data
ids = test_set['PassengerId']
predictions = classifier.predict(test_set.drop('PassengerId',axis=1))

output = pd.DataFrame({'PassengerId' : ids, 'Survived':predictions})
output.to_csv('titanic-predictions.csv', index = False)
output.head()


