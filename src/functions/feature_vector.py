# BAG OF WORDS

# In text processing, words of the text represent discrete, categorical features.

# How do we encode such data in a way which is ready to be used by the algorithms?

# The mapping from textual data to real valued vectors is called feature extraction.
# One of the simplest techniques to numerically represent text is Bag of Words.
# Bag of Words (BOW): We make the list of unique words in the text corpus called vocabulary.

# Then we can represent each sentence or document as a vector with each word represented as 1 for present and 0 for absent from the vocabulary.

from sklearn.feature_extraction.text import CountVectorizer #For Bag of words
from gensim.models import Word2Vec #For Word2Vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
import numpy as np

# Tokenize the collection of documents and form a vocabulary with it and use this vocabulary
# to encode new documents. We can use CountVectorizer of scikit-learn library.
# It by default remove punctuation’s and lower the documents.


# It turns each vector into sparse matrix.
# It will make sure the word present in the vocabulary and if present it prints
# the number of occurrences of the word in vocabulary.

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
def binary_count_vectorizer(preprocessor,untokenized_list):
    # create the transform
    # vectorizer = CountVectorizer(max_features=5000)
    vectorizer = CountVectorizer()
    # tokenize and build vocab
    vectorizer.fit(untokenized_list)
    #print(vectorizer.get_feature_names())
    #print(vectorizer.vocabulary_)
    # encode document
    vector = vectorizer.transform(untokenized_list)

    return vector

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
def bigram_count_vectorizer(preprocessor,untokenized_list):
    # create the transform
    # vectorizer = CountVectorizer(max_features=5000)
    vectorizer = CountVectorizer(ngram_range=(1,2))
    # tokenize and build vocab
    vectorizer.fit(untokenized_list)
    #print(vectorizer.get_feature_names())
    #print(vectorizer.vocabulary_)
    # encode document
    vector = vectorizer.transform(untokenized_list)

    return vector


# Word counts are pretty basic.
# Stop words can repeat several times in a document and word count prioritise with the occurrence of the word.
# From word counts, we loose the interesting words and we mostly give priority to stop words/less meaning carrying words.

# TF-IDF is popular method. Acronym is “Term Frequency and Inverse Document Frequency”.
# TF-IDF are word frequency scores that try to highlight words that are more interesting,
# e.g. frequent in a document but not across documents.

# For vector normalization, scikit-learn uses ‘l2’ normalization technique for each document.

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

# If you think that extremely high frequency may dominate the result and causing model bias.
# Normalization can be apply to pipeline easily.

def tfidf_vectorizer(preprocessor, untokenized_list):
    # create the transform
    vectorizer = TfidfVectorizer()
    # tokenize and build vocab
    vectorizer.fit(untokenized_list)
    #print(vectorizer.vocabulary_)
    #print('idfs: ', vectorizer.idf_)
    vector = vectorizer.transform(untokenized_list)
    return vector

#
# `size` is the dimensionality of the vector.
#
# Word2Vec needs large, varied text examples to create its 'dense' embedding vectors per word.

# (It's the competition between many contrasting examples during training which allows the word-vectors to move to

# positions that have interesting distances and spatial-relationships with each other.)
#
# If you only have a vocabulary of 30 words, word2vec is unlikely an appropriate technology.

# And if trying to apply it, you'd want to use a vector size much lower than your vocabulary size – ideally much lower.

# For example, texts containing many examples of each of tens-of-thousands of words might justify 100-dimensional word-vectors.
#
# Using a higher dimensionality than vocabulary size would more-or-less guarantee 'overfitting'.

# The training could tend toward an idiosyncratic vector for each word – essentially like a 'one-hot' encoding – that would perform better than any other encoding, because there's no cross-word interference forced by representing a larger number of words in a smaller number of dimensions.
#
# That'd mean a model that does about as well as possible on the Word2Vec internal nearby-word prediction task – but then awful on other downstream tasks, because there's been no generalizable relative-relations knowledge captured. (The cross-word interference is what the algorithm needs, over many training cycles, to incrementally settle into an arrangement where similar words must be similar in learned weights, and contrasting words different.)
#

def word2vec(preprocessor, untokenized_list):
    train_w2v = Word2Vec(untokenized_list,min_count=12,size=100, workers=4)
    avg_data = []
    for row in untokenized_list:
        vec = np.zeros(100)
        count = 0
        for word in row:
            try:
                vec += train_w2v[word]
                count += 1
            except:
                pass
        avg_data.append(vec/count)

    return avg_data


# V = ( t(W1)w2v(W1) + t(W2)w2v(W2) +.....+t(Wn)*w2v(Wn))/(t(W1)+t(W2)+....+t(Wn))

def word2vec_tfidf_vectorizer(word2vec_vector, untokenized_list):
    # create the transform
    vectorizer = TfidfVectorizer(max_features=5000)
    # tokenize and build vocab
    tf_idf_data = vectorizer.fit(untokenized_list)
    tf_idf_data = vectorizer.transform(untokenized_list)

    tf_w_data = []
    tf_idf_data = tf_idf_data.toarray()
    i = 0
    for row in untokenized_list:
        vec = [0 for i in range(100)]

        temp_tfidf = []
        for val in tf_idf_data[i]:
            if val != 0:
                temp_tfidf.append(val)

        count = 0
        tf_idf_sum = 0
        for word in row:
            try:
                count += 1
                tf_idf_sum = tf_idf_sum + temp_tfidf[count-1]
                vec += (temp_tfidf[count-1] * word2vec_vector[word])
            except:
                pass

        try:
            vec = (float)(1/tf_idf_sum) * np.array(vec)
        except ZeroDivisionError:
            vec = 0

        tf_w_data.append(vec)
        i = i + 1

    return tf_w_data


def custom_binary_bow_vector(vocabulary,docs):
    vectors=[]
    for doc in docs:
        vectors.append([1 if index in doc else 0 for index in vocabulary.keys()])
    return vectors


def custom_term_frequency_vector(top_words, final_documents):
     indexes_features = top_words.values()

     rows = []
     rows = indexes_features
     columns = []
     values = []

     # Term Frequency Matrix(tf)
     for val in final_documents:
         feature_vector = [0] * (len(indexes_features))
         for j in val:
             if j in rows:
                 feature_vector[rows.index(j)] = val.count(j)
         columns.append(feature_vector)
     vector = np.array(columns)
     return vector



def custom_tfidf_vector(top_words,final_documents):
    print("Generating tfidf vecor using tfidf transformer...")
    indexes_features = top_words.values()
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

    #Implementing tfidftransformer for creating tfidf_vector
    vectorizer = TfidfTransformer(norm=False,use_idf=True,sublinear_tf=True, smooth_idf=True)
    vectorizer.fit(columns)
    tfidf_vector = vectorizer.transform(columns)
    return tfidf_vector
