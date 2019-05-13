
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer #For Bag of words

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def build_model(mode,num_clusters):
    # Intent to use default paramaters for show case
    vect = None
    if mode == 'binary_count':
        vect = CountVectorizer()
    elif mode == 'bigram_count':
        vect = CountVectorizer(ngram_range=(1,2))
    elif mode == 'tf':
        vect = TfidfVectorizer(use_idf=False, norm='l2')
    elif mode == 'tfidf':
        vect = TfidfVectorizer()
    else:
        raise ValueError('Invalid model name..')

    return Pipeline([
        ('vect', vect),
        ('clf' , KMeans(num_clusters,random_state=99,init='k-means++', n_init=14, max_iter=100, tol=0.00001, copy_x=True))
    ])



def pipeline(x, y, mode):
    processed_x = preprocess_x(x)

    num_clusters = 2
    model_pipeline = build_model(mode,num_clusters)
    cv = KFold(n_splits=5, shuffle=True)

    scores = cross_val_score(model_pipeline, processed_x, y, cv=cv, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    return model_pipeline
