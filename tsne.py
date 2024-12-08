import matplotlib.pyplot as plt
import os

# transforms the data X to 2-D using TSNE then graphs labeled by Y
def display_tsne(X, Y):
    # transform data
    print(f'[TSNE] Transforming')
    X_tsne = _tsne(X)
    print(f'[TSNE] Dimension reduced to {X_tsne.shape[1]}')

    # plot data
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='viridis', s=50, edgecolor='k')
    plt.colorbar(scatter)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('2D t-SNE')

    plt.show()                      # for juypterlab/notebook
    plt.savefig('tsne.png')         # for terminal

def _cpu_tsne(X):
    tsne = TSNE(random_state=42)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def _gpu_tsne(X):
    tsne = TSNE(method='barnes_hut', perplexity=50, n_neighbors=150)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

if os.getenv("USEGPU"):
    from cuml.manifold import TSNE
    _tsne = _gpu_tsne
else:
    from sklearn.manifold import TSNE
    _tsne = _cpu_tsne

