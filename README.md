# 522Project
## Installing requirements
``requirements.txt`` contains all modules required for CPU only operation.

## GPU acceleration
Also install modules in ``requirements_gpu.txt`` (and do other pre-requisite GPU setup) for GPU acceleration.

GPU accleration isn't strictly necessary, but can be used to speed up training the *autoencoder* and *t-SNE* transformation.
The latter is benefits the most, as t-SNE calcluations are **very** slow, think kNN-scaling slow. If you want a t-SNE reduced representation of the data, I'd suggest decreasing the number of samples read in.

The Autoencoder should automatically detect the GPU once set up and utilize it.
t-SNE requires the ``USEGPU`` environment variable to be set to use GPU. 
This can be done with ``export USEGPU`` or with the use of a `.env` and the `dotenv` module. The latter is not implemented though.

## Pipeline
1. Load Data
2. Pre-process data
3. Dimensionality reduction (user selectable)
4. Classify data (user selectable)
5. Process data (classify)

Currently, post-processing is just merged in with the process module. It's fine for now since post-processing here is always the same: calculating AUC. 

## Dimensionality Reduction
1. Linear Discriminant Analysis (lda)
2. PCA (I dont remember the full name)
3. Autoencoder 

The autoencoder is currently reducing to an arbitrary dimension (51 I think). It isn't *completely* arbitrary though, 51 is around where PCA says atleast 95% of information is retained. 

## Processing
1. Decision Tree
2. LightBGM
3. RandomForest

## CLI Tool
main.py acts as a CLI program. It allows you to invoke any combination of the dimensionalaity reduction methods with any classifier (also referred to as processors in the code, referring to the "processing" step). 

## Where is the training data?
Training data is not kept in the repository since the files are too large. 

We also only use train.csv file to generate the training and validation sets.
We don't need sample_submission.csv or test.csv (though we could choose to do something with test.csv in the future).

The code expects it to be gzipped and named ``train.csv.gz``.
Compressing it saves storage space, pandas inflates it back up in the code.
Note that ``train.csv.gz`` is also already ignored by git so you don't accidentally push it up. 

We can also take this further and pickle the dataframe object direct, then gz that (like the MNIST dataset) but I'll test that at a later time (and its not as important as other things).

## Todo
We need to determine what we *actually* want our implementation to improve on. I assume  it's not going to be improvements on accuracy but improvements on dimensionality?

So far, I haven't systematically determined if PCA or the autoencoder is better. PCA, in general, just does a decent job. Autoencoding may have the potential to be even better, but since the autoencoder is itself a neural network, we'd have to find ideal hyperparameters.

If we do want to focus on the autoencoder for dimensionality reduction, it may be good to develop the autoencoder script into its own CLI for training the autoencoder specifically. 

There's only so many things we can mess with in the pipeline. 

Since we have context from the feature names and descriptions on Kaggle, we could try to improve the feature encodings.

Some categories are label encoded despite atleast having numbers (i.e. version numbers like "1.23.456.7"). We could use orginal encoders instead for these. Unlike label encoders, ordinal encoding can preserve the order of value (i.e [low, medium, high] should be encoded exactly to [0, 1, 2] in that order). I think there's also a alternate approach for version numbers where we could just strip the periods, parse the remaining digitsto numbers. 

There are some other categories that could benefit from the extra context of feature names and descriptions. But of course, this is a rather manual path to take. 
