# custom modules
from loader import load
from reduce import autoencode
from preprocess import fill_missing, display_missing, encode, standardize
from test import test_DTree, test_LGBM, test_RForest
from tsne import display_tsne

def main():
    X, Y = load('train.csv', nrows=1000000)

    # preprocess
    X = fill_missing(X)                                                                                         # fill NaNs
    X = encode(X)                                                                                               # categories -> numbers
    X = standardize(X)                                                                                          # standardization u=0, std=1

    # tests without dimensionality reduction
    test_DTree(X, Y, test_size=0.2)
    test_LGBM(X, Y, test_size=0.2)
    # test_RForest(X_scaled, Y, test_size=0.2)

    # Dimensionality reduction
    REDUCED_DIMS = 40
    X_low = autoencode(X, n_components=REDUCED_DIMS, save=True, force_refit=False, epochs=50, batch_size=256, shuffle=True)

    # test post-dimensionality reduction
    test_DTree(X_low, Y, test_size=0.2)
    test_LGBM(X_low, Y, test_size=0.2)


if __name__ == '__main__':
    main()