
from sklearn.cluster import KMeans

def kmeans(feature_vector, num_clusters):
    print "Clustering the matrix..."
    num_clusters = 2
    km = KMeans(num_clusters,random_state=99,init='k-means++', n_init=14, max_iter=100, tol=0.00001, copy_x=True)
    km.fit(feature_vector)
    clusters = km.labels_.tolist()
    return clusters
