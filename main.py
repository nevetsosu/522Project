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
    'pca': lambda X, Y=None: pca(X, n_components=0.95),
    'autoencode': lambda X, Y=None: autoencode(X, n_components=51),
}

PROCESSORS = {
    'decisiontree': DTree,
    'lgbm': LGBM,
    'randomforest': RForest
}

def pipeline(X, Y, test_size, preprocess: str, dim_reduce: str, process: str):
    if preprocess not in PREPROCESSORS:
        raise ValueError(f'{preprocess} is not a valid preprocessor\nChoose from: {PREPROCESSORS.keys()}' )
    elif dim_reduce not in DIM_REDUCTIONS:
        raise ValueError(f'{dim_reduce} is not a valid dimension reduction method\nChoose from: {DIM_REDUCTIONS.keys()}')
    elif process not in PROCESSORS:
        raise ValueError(f'{process} is not a valid processor\nChoose from: {PROCESSORS.keys()}')

    preprocessor = PREPROCESSORS[preprocess]
    dim_reducer = DIM_REDUCTIONS[dim_reduce]
    processor = PROCESSORS[process]

    print(f'Using preprocessor: {preprocess}')
    print(f'Using dimensionality reduction method: {dim_reduce}')
    print(f'Using processor: {process}')

    X = preprocessor(X=X)
    X = dim_reducer(X, Y)
    processor(X, Y, test_size=test_size)

def main():
    if len(sys.argv) != 3:
        print(f"{__file__} dim_reduce process")
        exit()

    # get parameters
    _, dim_reduce, process = sys.argv

    # load data
    TEST_SIZE = 0.2
    X, Y = load('train.csv', nrows=1000000)

    # construct and run pipeline
    pipeline(X, Y,
        test_size=TEST_SIZE,
        preprocess='default',
        dim_reduce=dim_reduce,
        process=process,
    )


if __name__ == '__main__':
    main()