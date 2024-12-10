from autoencoder import AutoEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def autoencode(X, n_components, **kwargs):
    return AutoEncoder(X.shape[1], n_components).fit_transform(X, **kwargs)

def pca(X, n_components, **kwargs):
    print(f'[REDUCE] Starting PCA')
    pca = PCA(n_components=n_components, **kwargs)
    X_pca = pca.fit_transform(X)
    print(f'[REDUCE] PCA reduced dimensionality to {X_pca.shape[1]}')
    return X_pca

def lda(X, Y, n_components=1, **kwargs):
    print(f'[REDUCE] Starting LDA')
    lda = LinearDiscriminantAnalysis(n_components=n_components, **kwargs)
    lda.fit(X, Y)
    X_lda = lda.transform(X)
    print(f'[REDUCE] LDA reduced dimensionality to {X_lda.shape[1]}')
    return X_lda


def cluster_reduce(X, n_components, linkage='ward', **kwargs):
    print(f'[REDUCE] Starting Agglomerative Hierarchical Clustering')
    X_T = X.T
    agglomerative = AgglomerativeClustering(n_clusters=n_components, linkage=linkage, **kwargs)
    features = agglomerative.fit_predict(X_T)

    print(f'[REDUCE] Agglomerative clustering grouped features into {n_components} clusters')
    X_cluster = np.zeros((X.shape[0], n_components))
    for cluster_id in range(n_components):
        cluster_features = X_T[features == cluster_id].T
        X_cluster[:, cluster_id] = np.mean(cluster_features, axis=1)
    
    print(f'[REDUCE] Clustering reduced dimensionality to {X_cluster.shape[1]}')
    return X_cluster