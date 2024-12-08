from sklearn.preprocessing import LabelEncoder, StandardScaler

# replace NaN with most frequent entry for categorical features
# replace NaN with median for numerical features
# this is an inplace operation but also returns df
def fill_missing(df):
    print('[PREPROCESS] Filling missing data')
    for col in df.columns:
        if df[col].dtype == 'category':
            df.fillna({col: df[col].mode()[0]}, inplace=True)
        else:
            df.fillna({col: df[col].median()}, inplace=True)
    print('[PREPROCESS] Done filling missing data')
    return df

# debug function for fill missing
# not used currently
def display_missing(df):
    print('[PREPROCESS] Displaying missing data:')
    for col in df.columns.tolist():
        num_null = df[col].isnull().sum()
        if num_null > 0:
            print(f'{col}: {num_null} missing values')
    print('[PREPROCESS] Done displaying missing data')

# encodes categories to numerical values
# this is an inplace operation but also returns df
def encode(df):
    print("[PREPROCESS] Label encoding categorical features")
    categorical = [col for col in df.columns if df[col].dtype == 'category']
    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    print('[PREPROCESS] Done label encoding categorical features')
    return df

# Transforms data so each feature has mean 0 and std 1
# NOT an inplace operation
def standardize(df):
    print("[PREPROCESS] Standardizing features")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    print('[PREPROCESS] Done standardizing features')
    return df_scaled

# fill missing -> encode -> standardize
def default_preprocess(X):
    X = fill_missing(X)
    X = encode(X)
    X = standardize(X)
    return X
