
import glob
import os
from nltk.tokenize import word_tokenize
import nltk.stem
import math

def load_data(directory):
    x = []
    y = []
    for f in glob.glob(os.path.join(directory,"HAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(0)
    for f in glob.glob(os.path.join(directory,"SPAM.*.txt")):
        with open( f, 'r')as file:
            x.append(file.read())
            y.append(1)
    return x,y


def create_index(corpus):
    inverted_index = {}
    sno=nltk.stem.SnowballStemmer('english')
    for doc_id, doc in enumerate(corpus):
        print(f'\r{round(((doc_id+1)/len(corpus))*100, 1) } % ',end='')
        for word in word_tokenize(doc):
            word=(word).upper()
            if word not in inverted_index:
               inverted_index[word] = 0
            inverted_index[word]+=1
    return inverted_index

def nb_train(x, y):
    model = {}
    ham = []
    spam = []
    for word, label in zip(x, y):
        if label == 0:
            ham.append(word)
        else:
            spam.append(word)
    ham_count=len(ham)
    spam_count=len(spam)
    model["ham_count"]=ham_count
    model["spam_count"]=spam_count
    ham_fd=create_index(ham)
    spam_fd=create_index(spam)
    model["ham_fd"]=ham_fd
    model["spam_fd"]=spam_fd
    return model

def spliting(docs):
    d=[]
    for doc in docs[:]:
        d.append(word_tokenize(doc))

    return d

def get_fd(Doc,ham,spam,Sumham,SumSpam,smooth,V):
    sno=nltk.stem.SnowballStemmer('english')
    listHam=[]
    listSpam=[]
    for W in Doc:
        Hamvalue=ham.get(W.upper())
        spamvalue=spam.get(W.upper())


        if (Hamvalue !=None)or(spamvalue !=None):
            if Hamvalue==None:
                Hamvalue=0
            if spamvalue==None:
                spamvalue=0
            listHam.append((Hamvalue+smooth)/(Sumham+V))
            listSpam.append((spamvalue+smooth)/(SumSpam+V))
    return listHam,listSpam


def multiply(list):
    value=1
    for i in list:
        if i==0:
            return 0
        value*=i
    return value

def useLog(list):
    value=0
    for i in list:
        if i==0:
            value += float('-inf')
            continue
        value+=math.log(i)
    return value
    
def nb_test(docs, trained_model, use_log = False, smoothing = False):
    result=[]
    smooth=0
    V=0
    N0=trained_model["ham_count"]
    N1=trained_model["spam_count"]
    Ntotal=N0+N1
    Pc0=N0/Ntotal
    Pc1=N1/Ntotal
    Sumham=sum(trained_model["ham_fd"].values())
    Sumspam=sum(trained_model["spam_fd"].values())
    Docs=spliting(docs)
    if(smoothing):
        V= len(set(list(model['ham_fd'].keys()) + list(model['spam_fd'].keys())))
        smooth=1

    for Doc in Docs:
        listHam,listSpam= get_fd(Doc,trained_model["ham_fd"],trained_model["spam_fd"],Sumham,Sumspam,smooth,V)
        if(use_log):   
            H=useLog(listHam)
            S=useLog(listSpam)
            H+=math.log(Pc0)
            S+=math.log(Pc1) 
        else:
            H=multiply(listHam)
            S=multiply(listSpam)
            H*=Pc0
            S*=Pc1

        if H<S:
            result.append(1)
        else:
            result.append(0)
    return result


def f_score(y_true, y_pred):
    true_positives = sum([1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1])
    false_positives = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1])
    false_negatives = sum([1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0])

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    F_score= 2 * (precision * recall) / (precision + recall)

    return F_score*100    

    


#===================================================================

print("hi 1")



x_train, y_train = load_data("./SPAM_training_set/")
model = nb_train(x_train, y_train)

x_test, y_test = load_data("./SPAM_test_set/")


y_pred = nb_test(x_test, model, use_log = False, smoothing = False)
print("use_log=f,smoothing=f",f_score(y_test,y_pred))

y_pred = nb_test(x_test, model, use_log = False, smoothing = True)
print("use_log=f,smoothing=t",f_score(y_test,y_pred))

y_pred = nb_test(x_test, model, use_log = True, smoothing = False)
print("use_log=t,smoothing=f",f_score(y_test,y_pred))

y_pred = nb_test(x_test, model, use_log = True, smoothing = True)
print("use_log=t,smoothing=t",f_score(y_test,y_pred))
