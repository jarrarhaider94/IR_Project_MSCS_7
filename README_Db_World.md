# Project Title

DB World Email Text Classification

## Getting Started

Project consists of 1 python files and 1 folders. "MATLAB" folder have all the text files in .mat format. Two datasets are stemmed and 2 are not stemmed. We have considered the stemmed dataset
### Prerequisites

Python 3.6 is required to run the code. All the necessary libraries need to be installed prior to running the code.

### Implementation

1 . Naive Bayes Classifier
2 . Rocchio Classifier
3 . KNN Classifier

### Implementation Details


As the dataset is structured and have proper labels and input features, therefore all we need is to store them in arrays. We provide these arrays into train_test_split function and divide the set into 25:75 ration of test and train sets.  We get X_Train, Y_Train, X_Text and Y_Test arrays. Then we feed these values to different classification algorithms provided by sklearn library. The code is given as follows:

```

db_world_matlab = scipy.io.loadmat('MATLAB/dbworld_bodies_stemmed.mat')



#Features/Labels in dbworld_bodies_stemmed file
labels = db_world_matlab['labels']
labels=np.asarray(labels)

#Input variable in dbworld_bodies_stemmed file
data = db_world_matlab['inputs']  
data=np.asarray(data)


#As we don't know the dimension, we just give provide the shape of label for first index.
#This way we don't have to look and insert the value manually.

X = pd.DataFrame(data)
y = labels.ravel()



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state = 39)

def Rocchio_Algorith():
    clf = NearestCentroid()
    clf.fit(X_train, Y_train)
    
    pred_result = clf.predict(X_test)

    print (pred_result)
    print()
    print (Y_test)

    print ('classification report: ')
#     print classification_report(y_test, yhat)
    print(classification_report(Y_test, pred_result))
    
    print ('f1 score')
    print (f1_score(Y_test, pred_result, average='macro'))
    
    print ('accuracy score')
    print (accuracy_score(Y_test, pred_result))
    
    precision = precision_score(Y_test, pred_result, average=None)
    print ("Precision : ")
    print (precision)
    recall = recall_score(Y_test, pred_result, average=None)
    print ("Recall : ")
    print (recall)

def Naive_Bayes_Algorith():
    
    mn = MultinomialNB()
    mn.fit(X_train, Y_train)

    pred_result = mn.predict(X_test)

    print (pred_result)
    print()
    print (Y_test)

    print ('classification report: ')
#     print classification_report(y_test, yhat)
    print(classification_report(Y_test, pred_result))
    
    print ('f1 score')
    print (f1_score(Y_test, pred_result, average='macro'))
    
    print ('accuracy score')
    print (accuracy_score(Y_test, pred_result))
    
    precision = precision_score(Y_test, pred_result, average=None)
    print ("Precision : ")
    print (precision)
    recall = recall_score(Y_test, pred_result, average=None)
    print ("Recall : ")
    print (recall)

def KNN_Algorithm():
    
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, Y_train)
    

    pred_result = knn.predict(X_test)

    print (pred_result)
    print()
    print (Y_test)

    print ('classification report: ')
#     print classification_report(y_test, yhat)
    print(classification_report(Y_test, pred_result))
    
    print ('f1 score')
    print (f1_score(Y_test, pred_result, average='macro'))
    
    print ('accuracy score')
    print (accuracy_score(Y_test, pred_result))
    
    precision = precision_score(Y_test, pred_result, average=None)
    print ("Precision : ")
    print (precision)
    recall = recall_score(Y_test, pred_result, average=None)
    print ("Recall : ")
    print (recall)

print ("_____Naive Bayes Classifier_____")
Naive_Bayes_Algorith()

print ("_____Rocchio Classifier_____")
Rocchio_Algorith()

print ("_____KNN Classifier_____")
KNN_Algorithm()


```

We calculate accuracy, score, precision and recall for all the three algorithms

### Steps for Running the Code

Open "db_word_classification.py" and just execute the code. Make sure the code is in the same folder as MATLAB is.


## Authors

Jarrar Haider
Ali Sheharyar
Sobas Mehboob




