from autoencoder import AutoEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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