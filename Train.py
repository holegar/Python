import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit, LeaveOneOut
import sys
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class Train:
    def __init__(self, Xtrn, Xtst, Ytrn, Ytst, param=None):
        self.__Xtrn = Xtrn
        self.__Xtst = Xtst
        self.__Ytrn = Ytrn
        self.__Ytst = Ytst
        self.__param = param

    def Forest(self):
        forest = RandomForestClassifier(n_estimators=self.__param)
        model = forest.fit(self.__Xtrn,self.__Ytrn) # Fit the model
        pred_train = model.predict(self.__Xtrn) # Make predictions on the test set
        pred_test = model.predict(self.__Xtst) # Make predictions on the test set
        cm = confusion_matrix(self.__Ytst,pred_test, normalize='true') # Confusion matrix
        cr = classification_report(self.__Ytst,pred_test)
        acc_train = accuracy_score(self.__Ytrn, pred_train) # Calculate the accuracy of train set
        acc_test = accuracy_score(self.__Ytst, pred_test) # Calculate the accuracy of test set
        cv = cross_val_score(model, self.__Xtrn, self.__Ytrn, cv = LeaveOneOut())
        return cm, cr, acc_train, acc_test, cv.mean()
    
    def SVM(self): 
        svc = SVC(kernel='linear', C=self.__param)
        model = svc.fit(self.__Xtrn, self.__Ytrn) # Fit the model
        pred_train = model.predict(self.__Xtrn) # Make predictions on the train set
        pred_test = model.predict(self.__Xtst) # Make predictions on the test set
        cm = confusion_matrix(self.__Ytst, pred_test, normalize='true') # Confusion matrix
        cr = classification_report(self.__Ytst,pred_test)
        acc_train = accuracy_score(self.__Ytrn, pred_train) # Calculate the accuracy of train set
        acc_test = accuracy_score(self.__Ytst, pred_test) # Calculate the accuracy of test set
        #cv = cross_val_score(model, self.__Xtrn, self.__Ytrn, cv = LeaveOneOut())
        return cm, cr, acc_train, acc_test#, cv.mean()
    
    def MLP(self): 
        mlp = MLPClassifier(random_state=1, max_iter = self.__param)
        model = mlp.fit(self.__Xtrn, self.__Ytrn) # Fit the model
        pred_train = model.predict(self.__Xtrn) # Make predictions on the train set
        pred_test = model.predict(self.__Xtst) # Make predictions on the test set
        cm = confusion_matrix(self.__Ytst, pred_test, normalize='true') # Confusion matrix
        cr = classification_report(self.__Ytst,pred_test)
        acc_train = accuracy_score(self.__Ytrn, pred_train) # Calculate the accuracy of train set
        acc_test = accuracy_score(self.__Ytst, pred_test) # Calculate the accuracy of test set
        #cv = cross_val_score(model, self.__Xtrn, self.__Ytrn, cv = LeaveOneOut())
        return cm, cr, acc_train, acc_test#, cv.mean()
    
    def KNN(self): 
        knn = KNeighborsClassifier(n_neighbors=self.__param)
        model = knn.fit(self.__Xtrn, self.__Ytrn) # Fit the model
        pred_train = model.predict(self.__Xtrn) # Make predictions on the train set
        pred_test = model.predict(self.__Xtst) # Make predictions on the test set
        cm = confusion_matrix(self.__Ytst, pred_test, normalize='true') # Confusion matrix
        cr = classification_report(self.__Ytst,pred_test)
        acc_train = accuracy_score(self.__Ytrn, pred_train) # Calculate the accuracy of train set
        acc_test = accuracy_score(self.__Ytst, pred_test) # Calculate the accuracy of test set
        #cv = cross_val_score(model, self.__Xtrn, self.__Ytrn, cv = LeaveOneOut())
        return cm, cr, acc_train, acc_test#, cv.mean()
    
    def Evaluate(self, cm, cr, acc_train, acc_test):#, cv):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        print("\nClassification Report:")
        print(cr)
        print("\nTrain set accuracy:", round(acc_train*100,2),"%")
        print("Test set accuracy:", round(acc_test*100,2),"%")
        #print("Cross validation accuracy:",round(cv*100,2),"%")