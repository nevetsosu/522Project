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
    dim_1 = X.shape[1]
    auc_1 = processor(X, Y, test_size=test_size)        # pre-dimension reduction results
    X = dim_reducer(X, Y)
    if show_tsne: display_tsne(X, Y)
    dim_2 = X.shape[1]
    auc_2 = processor(X, Y, test_size=test_size)        # post-dimension reduction results

    return (dim_1, auc_1), (dim_2, auc_2)

def fail():
    print(f"{__file__}  dim_reduce1 process1 [dim_reduce2] [process2]... [show_tsne(true/false)]")
    exit()

def main():
    show_tsne = False
    dim_methods = []
    processors = []
    all = False

    # get parameters
    argc = len(sys.argv)
    if argc == 1:
        fail()
    elif argc <= 3:                                       # handle "main all" and "main all true"
        if sys.argv[1].lower() == 'all':
            all = True
            if argc == 3:
                show_tsne = sys.argv[2] == 'true'
    elif not all and not (argc % 2):
        show_tsne = sys.argv[-1] == 'true'

    # configure pipeline combinations
    if not all:
        # get configurations from argv
        for i in range(1, argc - 1, 2):
            d = sys.argv[i]
            p = sys.argv[i + 1]
            if DIM_REDUCTIONS.get(d, None) is None:
                print(f'd is not a valid dimension reduction method')
                exit()
            if PROCESSORS.get(p, None) is None:
                print(f'p is not a valid processor')
                exit()

            dim_methods.append(sys.argv[i])
            processors.append(sys.argv[i + 1])
    else:
        # enumerate all combinations
        for dim_reduce in DIM_REDUCTIONS.keys():
            for process in PROCESSORS.keys():
                dim_methods.append(dim_reduce)
                processors.append(process)

    # display pipeline info
    print(f'-----[INFO START]-----')
    print(f'show t-SNE: {show_tsne}')
    print(f'Attempting {(len(dim_methods))} pipelines:')

    pipelines = list(zip(dim_methods, processors))
    for i, (dim_reduce, process) in enumerate(pipelines, 1):
        print(f"{i} |{dim_reduce} {process}|")

    print('------[INFO END]------')

    # load data
    X, Y = load('train.csv.gz', nrows=1000000, compression='gzip')

    TEST_SIZE = 0.2
    for i, (dim_reduce, process) in enumerate(pipelines, 1):
        print(f'[INFO] Starting Pipeline {i}: |{dim_reduce} {process}|')

        # construct and run pipeline
        (dim_1, auc_1), (dim_2, auc_2) = pipeline(X, Y,
            test_size=TEST_SIZE,
            preprocess='default',
            dim_reduce=dim_reduce,
            process=process,
            show_tsne=show_tsne,
        )

        print(f'[RESULT] full dim auc-score (dim={dim_1}): {auc_1}')
        print(f'[RESULT] reduced dim auc-score (dim={dim_2}): {auc_2}')
        print(f'[STAT] dim diff proportion: {(dim_2 - dim_1) / dim_1}')
        print(f'[STAT] auc diff proportion: {(auc_2 - auc_1) / auc_1}')
        print(f'[INFO] Pipeline {i} |{dim_reduce} {process}| finished.')


if __name__ == '__main__':
    main()