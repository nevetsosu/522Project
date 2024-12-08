import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# transforms the data X to 2-D using TSNE then graphs labeled by Y
def display_tsne(X, Y):
    COMPONENTS = 2

    # transform data
    print(f'[TSNE] Reducing dimensions to {COMPONENTS}')
    tsne = TSNE(n_components=COMPONENTS, random_state=42)
    X_tsne = tsne.fit_transform(X)
    print(f'[TSNE] Dimension reduced to {X_tsne.shape[1]}')

    # plot data
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='viridis', s=50, edgecolor='k')
    plt.colorbar(scatter)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('2D t-SNE')

    try:
        plt.show()                      # for juypterlab/notebook
    except:
        plt.savefig('tnse.png')         # for terminal
