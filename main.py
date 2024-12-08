# custom modules
from loader import load
from reduce import autoencode, pca, lda
from preprocess import fill_missing, display_missing, encode, standardize
from process import DTree, LGBM, RForest
from tsne import display_tsne

def main():
    TEST_SIZE = 0.2
    X, Y = load('train.csv', nrows=1000000)

    # preprocess
    X = fill_missing(X)         # fill NaNs
    X = encode(X)               # categories -> numbers
    X = standardize(X)          # standardization u=0, std=1

    # tests without dimensionality reduction
    DTree(X, Y, test_size=TEST_SIZE)
    LGBM(X, Y, test_size=TEST_SIZE)
    # test_RForest(X_scaled, Y, test_size=TEST_SIZE)

    # Dimensionality reduction
    # REDUCED_DIMS = 40
    # X = autoencode(X, n_components=REDUCED_DIMS, save=True, force_refit=False, epochs=50, batch_size=256, shuffle=True)
    # X = pca(X, n_components=0.95)
    X = lda(X, Y)

    # test post-dimensionality reduction
    DTree(X, Y, test_size=TEST_SIZE)
    LGBM(X, Y, test_size=TEST_SIZE)


if __name__ == '__main__':
    main()