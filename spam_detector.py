__author__   = "Emmanuvel Raphel"
__contact__   = "<raphelemmanuvel@gmail.com>"
__github__    = "https://github.com/emmanuvelraphel"

from sklearn.metrics import accuracy_score
import csv
import nltk
import re
from sklearn.cluster import KMeans
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer


#Reading the spam collection dataset
def reading_training_file():
    print "Reading messages from dataset..."
    documents = []
    with open("\dataset\sms_dataset.csv","rb") as csvfile:
        rows=csv.reader(csvfile)
        next(rows, None) #Skip column headers
        for row in rows:
            message = row[1]
            documents.append(message) #messages are appended to the list 'documents'
    print "Finished reading messages and appended to the list.."
    return documents

#Preprocessing steps including tokenization
def preprocessing(documents):
    print "Precprocessing the messages for clustering..."
    vocab_glob = {}
    tokenized_document = []
    final_documents=[]
    for document in documents:
        text=document.replace("</p>","") # removing </p>
        text=text.replace("<p>"," ")  # removing <p>
        text = text.replace("http", " ")
        text = text.replace("www", " ")
        text = re.sub(r'([a-z])\1+', r'\1', text)
        text = re.sub('\s+', ' ', text)
        text = re.sub('\.+', '.', text)
        text = re.sub(r"(?:\@|'|https?\://)\s+","",text) #delete punctuation
        text = re.sub("[^a-zA-Z]", " ",text)
        text=re.sub(r'[^\w\s]','',text) # remove punctuation
        text=re.sub("\d+","",text) # remove number from text
        tokens_text = nltk.word_tokenize(text) # tokenizing the documents
        stopwords=nltk.corpus.stopwords.words('english') #stopword reduction
        tokens_text=[w for w in tokens_text if w.lower() not in stopwords]
        tokens_text=[w.lower() for w in tokens_text] #convert to lower case
        tokens_text=[w for w in tokens_text if len(w)>2] #considering tokens with length>2(meaningful words)
        p= PorterStemmer() # stemming tokenized documents using Porter Stemmer
        tokens_text = [p.stem(w) for w in tokens_text]
        token_ind= []
        counter=len(vocab_glob)-1
        for token in tokens_text:
         if token not in vocab_glob:
            counter+=1
            vocab_glob[token]=counter
            token_ind.append(counter)
         else:
            token_ind.append(vocab_glob[token])
        final_documents.append(token_ind)
    print "Finished pre-processing words.."
    return vocab_glob,final_documents


#Feature Engineerng for selecting top words
def feature_selection(final_documents):
 print "Feature selection started..."
 doc_freq = {}
 for document in final_documents:
   for index in document:
    if index in doc_freq:
        doc_freq[index] += 1
    else:
        doc_freq[index] = 1
    top_features = []
    for token in doc_freq.keys():
     if doc_freq[token] >12:    #removing low frequent words(Band Pass Filtering)
      top_features.append(token)
 i = 0
 top_words = {}
 for token in top_features:
        top_words[i] = token
        i += 1
 return top_words


#Approach-1 --> Binary Document Representation(not employed for obtaining final accurate results)
def features_vector_binary(vocab_glob,final_documents):
    indexes_features=vocab_glob.values()
    rows=[]
    rows=indexes_features
    columns=[]
    values=[]

    #Term Frequency Matrix(tf)
    for val in final_documents:
        feature_vector = [0]*(len(indexes_features))
        for j in val:
         if j in rows:
             counter=1
             feature_vector[rows.index(j)] = counter
        columns.append(feature_vector)
    values=np.array(columns)
    return values



#Approach-2 --> Term Frequency Representaion(not employed for obtaining final accurate results)
def feature_matrix_term_frequency(top_words, final_documents):
 print final_documents
 indexes_features = top_words.values()
 print indexes_features

 rows = []
 rows = indexes_features
 columns = []
 values = []

 # Term Frequency Matrix(tf)
 for val in final_documents:
     feature_vector = [0] * (len(indexes_features))
     for j in val:
         if j in rows:
             counter = 1
             feature_vector[rows.index(j)] = val.count(j)
     columns.append(feature_vector)
 values = np.array(columns)
 print values.shape
 return values



#Approach-3 --> Creating the Term Frequency Matrix and Calculating tfidf weight using tfidf transformer - used for obtaining high accuracy
def feature_matrix_tfidf(top_words,final_documents):
    print "Calculating tfidf weight using tfidf transformer..."
    #print final_documents
    indexes_features = top_words.values()
    #print indexes_features
    rows = []
    rows = indexes_features
    columns = []
    values = []

    for val in final_documents:
        feature_vector = [0] * (len(indexes_features))
        for j in val:
            if j in rows:
                feature_vector[rows.index(j)] = val.count(j)
        columns.append(feature_vector)
    #print columns
    #values = np.array(columns)
    #print values.shape
    #return columns

    #Implementing tfidftransformer for creating TfidTransformer from scikit-learn
    tfidf = TfidfTransformer(norm=False,use_idf=True,sublinear_tf=True, smooth_idf=True)
    tfidf.fit(columns)
    tfidf_matrix = tfidf.transform(columns)
    test=tfidf_matrix
    #print tf_idf_matrix.todense()
    return test



#Clustering the tfidf matrix using k-means algorithm from scikit learn
def kmeans(f_vector):
    print "Clustering the matrix..."
    num_clusters = 2
    km = KMeans(num_clusters,random_state=99,init='k-means++', n_init=14, max_iter=100, tol=0.00001, copy_x=True)
    km.fit(f_vector)
    clusters = km.labels_.tolist()
    print "Results of Clustering:"
    print clusters
    print "Length of results:"
    print len(clusters)
    return clusters

#Printing Clustering Results to the file with headers and serial_no
def write_to_csv(clusters):
   print "opening the file to save clustering results....."
   file_save = "\results\final_results.csv"
   with open(file_save, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(["SMS_id", "label"])
    numbers = list(range(1, len(clusters)+1))
    print "writing to the file...."
    for row in zip(numbers, clusters):
     writer.writerow(row)
    print "Completed writing process"

   #printing the clustering results to a different file without headers to calculate the accuracy by comparing with 'original_accurate_results' file
   f_save = "\results\clustering_results.csv"
   with open(f_save, 'wb') as f:
    writer = csv.writer(f)
    for row in clusters:
        writer.writerow([row])

#Calculating accuracy of results by comparing with original results
#In the original_accurate_results file ham mesages are denoted by '1' and spam messages by '0'
#The k-means clustering results will be produce random results for different runs(ie, sometimes 1 for ham and 0 for spam and the vice versa)
#So make sure that we are comparing the results file having results --> ham as '1' and spam as '0' with original_accurate_results file, by converting the results appropriately
def read_accuracy_file():
    list_prediction=[]
    list_true = []
    #clustering_results
    with open("\results\clustering_results.csv","rb") as f_pre:
        rows= csv.reader(f_pre)
        for row in rows:
            if row[0]==str(0):
                list_prediction.append(int(row[0]))
            elif row[0]==str(1):
                list_prediction.append(int(row[0]))

    with open("\results\accurate_results.csv","rb") as f:
        rows= csv.reader(f)
        for row in rows:
            if row[0]==str(0):
                list_true.append(0)
            elif row[0]==str(1):
                list_true.append(1)
    return list_prediction, list_true


#Main function
def main():
    documents = reading_training_file()
    vocab_glob,final_documents=preprocessing(documents)
    top_words= feature_selection(final_documents)
    #feature_vector= features_vector_binary(vocab_glob,final_documents)
    #freq_matrix=feature_matrix_term_frequency(top_words, final_documents)
    feature_vector=feature_matrix_tfidf(top_words, final_documents)
    clusters=kmeans(feature_vector)
    write_to_csv(clusters)
    y_pre, list_true = read_accuracy_file()
    accuracy =  accuracy_score(list_true, y_pre)
    print ("Accuracy score:", accuracy)
main()
