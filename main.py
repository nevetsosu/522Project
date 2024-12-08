import sys

# custom modules
from loader import load
from reduce import autoencode, pca, lda
from preprocess import default_preprocess
from process import DTree, LGBM, RForest
from tsne import display_tsne

PREPROCESSORS = {
    'default': default_preprocess,
}

DIM_REDUCTIONS = {
    'lda': lda,
    'pca': lambda X, _: pca(X, n_components=0.95),                 # n_components is set arbitrarily for now
    'autoencoder': lambda X, _: autoencode(X, n_components=51, save=True),     # n_components is set arbitrarily for now
}

PROCESSORS = {
    'decisiontree': DTree,
    'lgbm': LGBM,
    'randomforest': RForest
}

def pipeline(X, Y, test_size, preprocess: str, dim_reduce: str, process: str, show_tsne=False):
    if preprocess not in PREPROCESSORS:
        raise ValueError(f'{preprocess} is not a valid preprocessor\nChoose from: {list(PREPROCESSORS.keys())}' )
    elif dim_reduce not in DIM_REDUCTIONS:
        raise ValueError(f'{dim_reduce} is not a valid dimension reduction method\nChoose from: {list(DIM_REDUCTIONS.keys())}')
    elif process not in PROCESSORS:
        raise ValueError(f'{process} is not a valid processor\nChoose from: {list(PROCESSORS.keys())}')

    # assign pipeline modules
    preprocessor = PREPROCESSORS[preprocess]
    dim_reducer = DIM_REDUCTIONS[dim_reduce]
    processor = PROCESSORS[process]

    # debug message
    print(f'[PIPELINE] Using preprocessor: {preprocess}')
    print(f'[PIPELINE] Using dimensionality reduction method: {dim_reduce}')
    print(f'[PIPELINE] Using processor: {process}')

    # run pipeline
    X = preprocessor(X)
    processor(X, Y, test_size=test_size)        # pre-dimension reduction results
    X = dim_reducer(X, Y)
    if show_tsne: display_tsne(X, Y)
    processor(X, Y, test_size=test_size)        # post-dimension reduction results

def main():
    show_tsne = False

    # get parameters
    if len(sys.argv) == 3:
        _, dim_reduce, process = sys.argv
    elif len(sys.argv) == 4:
        _, dim_reduce, process, show_tsne = sys.argv
        show_tsne = show_tsne.lower() == 'true'
    else:
        print(f"{__file__} dim_reduce process")
        exit()

    # load data
    X, Y = load('train.csv', nrows=1000000)

    # construct and run pipeline
    TEST_SIZE = 0.2
    pipeline(X, Y,
        test_size=TEST_SIZE,
        preprocess='default',
        dim_reduce=dim_reduce,
        process=process,
        show_tsne=show_tsne,
    )


if __name__ == '__main__':
    main()