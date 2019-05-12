from sklearn.feature_extraction.text import CountVectorizer #For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer #For TF-IDF
import nltk

def get_vocabulary(tokens_list):

    vocabulary = {}
    vectorized_list=[]

    counter=len(vocabulary)-1

    for list in tokens_list:
        vector_index= []
        for token in list:
            if token not in vocabulary:
                counter+=1
                vocabulary[token]=counter
                vector_index.append(counter)
            else:
                vector_index.append(vocabulary[token])
        vectorized_list.append(vector_index)
    return vocabulary, vectorized_list



def get_word_frequency(vecorized_list):
     print("Calculate frequencies of each word in the vocabulary...")

     word_frequency = {}

     for vector in vecorized_list:
       for index in vector:
         if index in word_frequency:
            word_frequency[index] += 1
         else:
            word_frequency[index] = 1
     print("Finished calculating frequency distribution...")
     return word_frequency


def get_top_words(freq_dist, threshold_value):
       top_features = []
       for freq in freq_dist.keys():
         if freq_dist[freq] > threshold_value:    #removing low frequent words(Band Pass Filtering)
          top_features.append(freq)
       i = 0
       top_words = {}
       for token in top_features:
         top_words[i] = token
         i += 1
       return top_words


def get_key(dictionary,value):
    for k in dictionary.keys():
        if (dictionary[k] == value):
            return k
