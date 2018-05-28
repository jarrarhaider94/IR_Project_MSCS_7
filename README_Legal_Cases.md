# Project Title

Legal Cases Text Classification

## Getting Started

Project consists of two python files and 5 folders. "citations_class" folder have all the text files in .xml format. "testing" and "training" folders have the data that has been split into ration 30/70. "training_preprocessed" and "testing_preprocessed" have all the files after stop-words removal.

### Prerequisites

Python 3.6 is required to run the code. All the necessary libraries need to be installed prior to running the code.

### Implementation
1 . Naive Bayes Classifier
2 . Rocchio Classifier
3 . KNN Classifier

### Implementation Details

First of all, we run the "train_test_split.py", if our trainng, testing, training_preprocessed and testing_preprocessed folders are empty. This code gets the files from the given folder and split then in 30:70 ratio. Once we have all the set of files belonging to training and testing arrays, we copy those files to their respective folders. With each file that is copied to train or test folders, we fix the issues in .xml file for it to be readable. Once this is done, we have to remove the stop-words. This is done using nlkt library and using the stop-words provided there.
Preprocessing is applied on each file from training and testing. The resulting file is save to new folders.
The code is given as follows:

```
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.xml'), all_files))
    return data_files

from random import shuffle

def randomize_files(file_list):
    shuffle(file_list)
    


def get_training_and_testing_sets(file_list):
    split = 0.7
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

data_files = get_file_list_from_dir("citations_class")

randomize_files(data_files)

training,testing  = get_training_and_testing_sets(data_files)

for training_files in training:
    copyfile('citations_class/' + training_files , "training/" + training_files)
    with in_place.InPlaceText('training/' + training_files , encoding = 'cp1252') as file:
        for line in file:
            line = line.replace('"id=', 'id="')
            line = line.replace('&','')
            file.write(line)
for testing_files in testing:
    copyfile('citations_class/' + testing_files , "testing/" + testing_files)
    with in_place.InPlaceText('testing/' + testing_files , encoding = 'cp1252') as file:
        for line in file:
            line = line.replace('"id=', 'id="')
            line = line.replace('&','')
            file.write(line)
    
    
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.xml'), all_files))
    return data_files

data_files = get_file_list_from_dir("training")

for file in data_files:
    with codecs.open('training/'+file,'r',encoding='cp1252') as inFile, codecs.open('training_preprocessed/'+file,'w',encoding='cp1252') as outFile:
        for line in inFile.readlines():
            print(" ".join([word for word in line.lower().translate(str.maketrans('', '', '')).split() 
            if len(word) >=4 and word not in stopwords.words('english')]), file=outFile)


data_files = get_file_list_from_dir("testing")
    
for file in data_files:
    with codecs.open('testing/'+file,'r',encoding='cp1252') as inFile, codecs.open('testing_preprocessed/'+file,'w',encoding='cp1252') as outFile:
        for line in inFile.readlines():
            print(" ".join([word for word in line.lower().translate(str.maketrans('', '', '')).split() 
            if len(word) >=4 and word not in stopwords.words('english')]), file=outFile)


```

Next, we execute the file "Legal_Case_Classificatio.py". First function we define is "get_file_list_from_dir" that takes folder/directory name as parameter and return all the names of files in it. The code is given as follows:

```
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.xml'), all_files))
    return data_files
```

Next, we define a function "get_all_classes_and_labels_from_testing" that takes name of file and parses the xml and gets the name of class and label for that class and saves it into arrays. The code is given as follows:

```

def get_all_classes_and_labels_from_testing(file_path):
    tree = ET.parse('training_preprocessed/' + file_path)
    root = tree.getroot()
    for class_label in root.iter('class'):
        class_label_array.append(class_label.text)
    for text_label in root.iter('text'):
            text_.append(text_label.text)

```

Next, we define a function "list_files_from_directory" that takes directory/folder name as input and gives the names of files in that folder. This is being used to fetch files from "testing_preprocessed" folder. The code is given as follows:

```

def list_files_from_directory(directory):
    """Lists all file paths from given directory"""

    ret_val = []
    for file in os.listdir(directory):
        if file.endswith(".xml"):
            ret_val.append(str(directory) + "/" + str(file))
    return ret_val


```

Next, we define a function "prepare_test_data" that takes name of folder as input and parses over the files in that folder. This is done by calling "list_files_from_directory" function. Once we have all the names of files in a folder, we parse over the xml file and get labels and text for testing and save it in respective arrays. The code is given as follows:

```

ef prepare_test_data(input_folder):
    """Maps each sentence to it's category"""

    test_folder = list_files_from_directory(input_folder)
    t_sentences = []
    t_categories = []
    
    
    for file in test_folder:
        tree = ET.parse(file)
        root = tree.getroot()
        for class_label in root.iter('class'):
            t_categories.append(class_label.text)
        for text_label in root.iter('text'):
            t_sentences.append(text_label.text)
            
    return t_categories, t_sentences


```

Next, we define a function "create_text_label_array" that gives appends all the labels in one array. The code is given as follows:

```
def create_text_label_array():
#        
    for x in range(len(text_)):
        training_label.append(class_label_array[x])
        training_sentences.append(text_[x])

```

Next, we start to call the functions to get training and testing data. The code is given as follows:

```
all_files = get_file_list_from_dir("training_preprocessed")
for files in all_files:
    get_all_classes_and_labels_from_testing(files)
    #print(files)
    
create_text_label_array()
categories, sentences = prepare_test_data("testing_preprocessed")

```

Next, there were some issues in the xml. There were some indices in text array, for both training and testing, that had "None" value. This created issues when training. To remove those and the class labels agains't those indices, the following code is written. This code find the index where the value is "None" and remove the corresponding class. Then filter is applied to remove elements that have "None" value:

```
x = 0
for s in training_sentences:
    if s is None:
        del training_label[x]
    x = x+1


training_sentences = list(filter(None, training_sentences))

z = 0
for sen in sentences:
    if sen is None:
        del categories[z]
    z = z + 1
    
sentences = list(filter(None, sentences))


```


As we need to have documents in matrix format for classification, we implement the following function to convert the text arrays into vectorised matrix format. Once we have vector format, we transform this vector into tf-idf form.

```
#Create text vectorizer
vectorizer = CountVectorizer()
vectorizer.fit(training_sentences)
train_mat = vectorizer.transform(training_sentences)
test_mat = vectorizer.transform(sentences)

#Applying TF-IDF to the vector matrix 
tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)
print (train_tfmat.shape)
test_tfmat = tfidf.transform(test_mat)
print (test_tfmat.shape)
```




Next, we define a function testClassifier. This function first trains the classifier and then tests it of the given test array. This function takes 5 parameters. 
1. x_train
2. y_train
3. x_text
4. y_test
5. clf (classifier object for sklearn library)

It outputs recall, precision, accuracy, f1-score and confusion matrix.

The code is given as follows:
```
def testClassifier(x_train, y_train, x_test, y_test, clf):
    
    
    metrics = []
    start = dt.now()
    clf.fit(x_train, y_train)
    end = dt.now()
    print ('training time: ', (end - start))
    
    # add training time to metrics
    metrics.append(end-start)
    
    start = dt.now()
    yhat = clf.predict(x_test)
    end = dt.now()
    print ('testing time: ', (end - start))
    
    # add testing time to metrics
    metrics.append(end-start)
    
    print ('classification report: ')
#     print classification_report(y_test, yhat)
    pp.pprint(classification_report(y_test, yhat))
    
    print ('f1 score')
    print (f1_score(y_test, yhat, average='macro'))
    
    print ('accuracy score')
    print (accuracy_score(y_test, yhat))
    
    precision = precision_score(y_test, yhat, average=None)
    recall = recall_score(y_test, yhat, average=None)
    
    # add precision and recall values to metrics
    for p, r in zip(precision, recall):
        metrics.append(p)
        metrics.append(r)
    
    
    #add macro-averaged F1 score to metrics
    metrics.append(f1_score(y_test, yhat, average='macro'))
    
    print ('confusion matrix:')
    print (confusion_matrix(y_test, yhat))
    
    # plotting the confusion matrix
    plt.imshow(confusion_matrix(y_test, yhat), interpolation='nearest')
    plt.show()
    
    return metrics

```



Next, we call 3 different classifiers to train on the them and then perform test. The code for the classifiers is given as follows:

```
metrics_dict = []
# Bayes Classifier

mbn_params = {'alpha': [a*0.1 for a in range(0,11)]}
mbn_clf = GridSearchCV(MultinomialNB(), mbn_params, cv=10)
mbn_clf.fit(train_tfmat, training_label)
print ('best parameters')
print (mbn_clf.best_params_)
best_mbn = MultinomialNB(alpha=mbn_clf.best_params_['alpha'])
best_mbn_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, best_mbn)
metrics_dict.append({'name':'Best MultinomialNB', 'metrics':best_mbn_me})


# KNN
for nn in [10]:
    print ('knn with ', nn, ' neighbors')
    knn = KNeighborsClassifier(n_neighbors=nn)
    knn_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, knn)
    metrics_dict.append({'name':'5NN', 'metrics':knn_me})
    print (' ')


#Rocchio Algorithm
    
clf = NearestCentroid()
clf_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, clf)
metrics_dict.append({'name':'Rocchio Algorithm', 'metrics':clf_me})


```

The results can be seen once the code is executed


### Steps for Running the Code

First of all, all the libraries included in the code need to be installed.
We start with executing, train_test_split.py if "training" and "testing" folders are empty. This code get the names of all the files in "citations_class" folder and split them into training and testing groups. These files are then stored into their respective folders. If you want to create a new train and test split, delete all the files from "training" and "testing" folder and then run "train_tes_split.py".

Once we have the "training_preprocessed" and "testing_preprocessed" datasets, we execute "Legal_Case_Classification". 


## Authors

Jarrar Haider
Ali Sheharyar
Sobas Mehboob


## Acknowledgments

* Github user @vdragan1993 


