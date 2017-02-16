from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier

"""
4 types of Classifiers:
0. Decission Trees
1. Suport Vector
2. Guassian naive_bayes
3. K Nearest Neighbours
4. Neural Network

"""
clf0 = tree.DecisionTreeClassifier()
clf1 = svm.SVC()
clf2 = GaussianNB()
clf3 = neighbors.KNeighborsClassifier(n_neighbors = 6)
clf4 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 3), random_state=1)

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf0 = clf0.fit(X, Y)
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)
clf4 = clf4.fit(X,Y)

#test data
X_test=[[198,92,48],[184,84,44],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
Y_test=['male','male','male','female','female','female','male','male']

Y_prediction0 = clf0.predict(X_test)
Y_prediction1 = clf1.predict(X_test)
Y_prediction2 = clf2.predict(X_test)
Y_prediction3 = clf3.predict(X_test)
Y_prediction4 = clf4.predict(X_test)

#Comparing the Results
print("Accuracy for Desiccion Trees : ",accuracy_score(Y_test,Y_prediction0))
print("Accuracy for SVM : ",accuracy_score(Y_test,Y_prediction1))
print("Accuracy for Naive Bayes : ",accuracy_score(Y_test,Y_prediction2))
print("Accuracy for K neighbors : ",accuracy_score(Y_test,Y_prediction3))
print("Accuracy for Neural Network : ",accuracy_score(Y_test,Y_prediction4))
