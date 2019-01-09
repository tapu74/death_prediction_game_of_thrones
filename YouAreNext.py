# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing


data ="character-deaths.csv"
names = ['Name', 'Allegiances', 'Death_Year', 'Book_of_Death', 'Death_Chapter','Book_Intro_Chapter','Gender','Nobility','GoT','Cok','SoS','FfC','DwD']
datasetMain = pandas.read_csv(data)


###We can get a quick idea of how many instances (rows)
### and how many attributes (columns) the data contains with the shape property.####

#droping column which are not necessary

to_drop = ['Name','Death Year','Allegiances','Book of Death','Death Chapter']
datasetMain.drop(to_drop,inplace=True, axis=1)

#not consider rows with no death
#pandas.notnull(dataset['Death Year'])
#dataset.dropna(subset=[2])
#dataset = datasetMain[pandas.notnull(datasetMain['Death Year'])]

#aliveDataset = datasetMain[pandas.isnull(datasetMain['Death Year'])]
dataset = datasetMain

# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())


# class distribution
print(dataset.groupby('AllegiancesDigit').size())




# Split-out validation dataset
array = dataset.values
X = array[:,1:10]
Y = array[:,0]

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)



# Test options and evaluation metric
seed = 7
scoring = 'accuracy'




# Spot Check Algorithms
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LR', LogisticRegression()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('CART', DecisionTreeClassifier()))


# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)



# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()




# Make predictions on validation dataset with Linear Discriminant Analysis (LDA) model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)


predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#### Here is some character data, now predict ;) ####
#Jaime Lannister 13,5,1,1,1,1,1,1,1
#Arya Strak 17,2,0,1,1,1,1,1,1
#Cersei Lannister 7,4,0,1,1,1,1,1,1
#Jon Snow 15,1,1,1,1,1,1,1,1
#Daenerys Targaryen 10,3,0,1,1,1,1,0,1
#Dead 1,28,1,0,1,0,1,0,0
#


predictionsNew = lda.predict([[13,5,1,1,1,1,1,1,1],
                              [17,2,0,1,1,1,1,1,1],
                              [7,4,0,1,1,1,1,1,1],
                              [1,28,1,0,1,0,1,0,0],
                              [15,1,1,1,1,1,1,1,1],
                              [10,3,0,1,1,1,1,0,1],
                              ])

print(predictionsNew)


