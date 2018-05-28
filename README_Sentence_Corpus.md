# Project Title

Sentence Corpus Text Classification

## Getting Started

Project consists of two python files and 4 folders. "Labeled_articles" folder have all the text files in .txt format. "testing" and "training" folders have the data that has been split into ration 30/70.

### Prerequisites

Python 3.6 is required to run the code. All the necessary libraries need to be installed prior to running the code.

### Implementation
1 . Naive Bayes Classifier
2 . Rocchio Classifier
3 . KNN Classifier

### Implementation Details


First of all, we run the code "train_test_split.py" if training and testing folders are empty. This code gets the files from the given folder and split then in 30:70 ratio. Once we have all the set of files belonging to training and testing arrays, we copy those files to their respective folders. The code is given as follows:

```
def get_file_list_from_dir(datadir):
all_files = os.listdir(os.path.abspath(datadir))
data_files = list(filter(lambda file: file.endswith('.txt'), all_files))
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

data_files = get_file_list_from_dir("labeled_articles")

randomize_files(data_files)

training,testing  = get_training_and_testing_sets(data_files)

for training_files in training:
copyfile('labeled_articles/' + training_files , "training/" + training_files)

for testing_files in testing:
copyfile('labeled_articles/' + testing_files , "testing/" + testing_files)

```

In sentence classification.py, first we define the function defined is "list_files_from_directory" that takes folder/directory name as parameter and return all the names of files in it. The code is given as follows:

```
def list_files_from_directory()
"""Lists all file paths from given directory"""

    ret_val = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            ret_val.append(str(directory) + "/" + str(file))
    return ret_val
```

Next, we define the function "read_file", that is used to to get all the lines in a given file except those lines that stars with #. This function takes path of file as parameter. The code is given as follows:
```
def read_file(path):
    """Reads all lines from file on given path"""

    f = open(path, "r")
    read = f.readlines()
    ret_val = []
    for line in read:
        if line.startswith("#"):
            pass
        else:
            ret_val.append(line)
    return ret_val


```


Next, we define a function "process_line" that is used to remove all the stop-words in the given text and return the labels for each sentence. This function takes a line or a string from the file as parameter. The code is given as follows:
```
def process_line(line):
    """Returns sentence category and sentence in given line"""

    if "\t" in line:
        splits = line.split("\t")
        s_category = splits[0]
        sentence = splits[1].lower()
        for sw in stopwords:
            sentence = sentence.replace(sw, "")
        pattern = re.compile("[^\w']")
        sentence = pattern.sub(' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return s_category, sentence
    else:
        splits = line.split(" ")
        s_category = splits[0]
        sentence = line[len(s_category)+1:].lower()
        for sw in stopwords:
            sentence = sentence.replace(sw, "")
        pattern = re.compile("[^\w']")
        sentence = pattern.sub(' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return s_category, sentence
```

Next, we define a a function create_sentence_label_array. In this function, we are storing all the text and its labels in form of array so that we could get a we could use them for training. Function takes directory/folder name as parameter. The code is given as follows:
```
def create_sentence_label_array(input_folder):
    """Writes training data from given folder into formatted JSON file"""

    tr_folder = list_files_from_directory(input_folder)
    for file in tr_folder:
        lines = read_file(file)
        for line in lines:
            c, s = process_line(line)
            if s.endswith('\n'):
                s = s[:-1]
            training_label.append(c)
            training_sentences.append(s)

```

Next, we define a function prepare_test_data. This function gets all the text and labels from the testing folder and saves it in respective arrays. Function takes directory/folder name as parameter. The code is given as follows:
```
def prepare_test_data(input_folder):
    """Maps each sentence to it's category"""

    test_folder = list_files_from_directory(input_folder)
    t_sentences = []
    t_categories = []
    for file in test_folder:
        lines = read_file(file)
        for line in lines:
            c, s = process_line(line)
            if s.endswith('\n'):
                s = s[:-1]
            t_sentences.append(s)
            t_categories.append(c)
    return t_categories, t_sentences

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


Now, we start to call the functions. First of all, we remove the stop-words. List of stop-words is provided in "word_list" folder. The code is given as follows:

```
# loading stopwords
input_stopwords = read_file("word_lists/stopwords.txt")
stopwords = []
for word in input_stopwords:
    if word.endswith('\n'):
        word = word[:-1]
        stopwords.append(word)

```

Next, we call the functions to make train and test text and labels respectively.

```
# prepare training and test data
create_sentence_label_array("training")
categories, sentences = prepare_test_data("testing")
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

Next, we call 3 different classifiers to train on the them and then perform test. The code for the classifiers is given as follows:

```

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
for nn in [3]:
    print ('knn with ', nn, ' neighbors')
    knn = KNeighborsClassifier(n_neighbors=nn)
    knn_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, knn)
    metrics_dict.append({'name':'3NN', 'metrics':knn_me})
    print (' ')


#Rocchio Algorithm
    
clf = NearestCentroid()
clf_me = testClassifier(train_tfmat, training_label, test_tfmat, categories, clf)
metrics_dict.append({'name':'Rocchio Algorithm', 'metrics':clf_me})


```

The results can be seen once the code is executed


### Steps for Running the Code

First of all, all the libraries included in the code need to be installed.
We start with executing, train_test_split.py if "training" and "testing" folders are empty. This code get the names of all the files in "labeled_articles" folder and split them into training and testing groups. These files are then stored into their respective folders. If you want to create a new train and test split, delete all the files from "training" and "testing" folder and then run "train_tes_split.py".

Once we have the "training" and "testing" datasets, we execute "Sentense Classification.py". 


## Authors

Jarrar Haider
Ali Sheharyar
Sobas Mehboob


## Acknowledgments

* Github user @vdragan1993 


