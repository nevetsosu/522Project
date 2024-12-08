import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def display_missing(df):
    print('Missing values in training data:')
    for col in df.columns.tolist():
        num_null = df[col].isnull().sum()
        if num_null > 0:
            print(f'{col}: {num_null} missing values')

def fill_missing(df):
    for col in df.columns:
        if df[col].dtype == 'category':
            df.fillna({col: df[col].mode()[0]}, inplace=True)
        else:
            df.fillna({col: df[col].median()}, inplace=True)

def display_tSNE(X):
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(X)

    plt.scatter(x_tsne[:, 0], x_tsne[:, 1])
    plt.title('t-SNE Projection')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

def main():
    INDEX='MachineIdentifier'
    LABELS='HasDetections'

    dtypes = {
            'MachineIdentifier':                                    'category',
            'ProductName':                                          'category',
            'EngineVersion':                                        'category',
            'AppVersion':                                           'category',
            'AvSigVersion':                                         'category',
            'IsBeta':                                               'Int8',
            'RtpStateBitfield':                                     'float32',
            'IsSxsPassiveMode':                                     'Int8',
            'DefaultBrowsersIdentifier':                            'float32',
            'AVProductStatesIdentifier':                            'float32',
            'AVProductsInstalled':                                  'float32',
            'AVProductsEnabled':                                    'float32',
            'HasTpm':                                               'Int8',
            'CountryIdentifier':                                    'int32',
            'CityIdentifier':                                       'float32',
            'OrganizationIdentifier':                               'float32',
            'GeoNameIdentifier':                                    'float32',
            'LocaleEnglishNameIdentifier':                          'int32',
            'Platform':                                             'category',
            'Processor':                                            'category',
            'OsVer':                                                'category',
            'OsBuild':                                              'int16',
            'OsSuite':                                              'int16',
            'OsPlatformSubRelease':                                 'category',
            'OsBuildLab':                                           'category',
            'SkuEdition':                                           'category',
            'IsProtected':                                          'float32',
            'AutoSampleOptIn':                                      'Int8',
            'PuaMode':                                              'category',
            'SMode':                                                'float32',
            'IeVerIdentifier':                                      'float32',
            'SmartScreen':                                          'category',
            'Firewall':                                             'float32',
            'UacLuaenable':                                         'float64',
            'Census_MDC2FormFactor':                                'category',
            'Census_DeviceFamily':                                  'category',
            'Census_OEMNameIdentifier':                             'float32',
            'Census_OEMModelIdentifier':                            'float32',
            'Census_ProcessorCoreCount':                            'float32',
            'Census_ProcessorManufacturerIdentifier':               'float32',
            'Census_ProcessorModelIdentifier':                      'float32',
            'Census_ProcessorClass':                                'category',
            'Census_PrimaryDiskTotalCapacity':                      'float32',
            'Census_PrimaryDiskTypeName':                           'category',
            'Census_SystemVolumeTotalCapacity':                     'float32',
            'Census_HasOpticalDiskDrive':                           'Int8',
            'Census_TotalPhysicalRAM':                              'float32',
            'Census_ChassisTypeName':                               'category',
            'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float32',
            'Census_InternalPrimaryDisplayResolutionHorizontal':    'float32',
            'Census_InternalPrimaryDisplayResolutionVertical':      'float32',
            'Census_PowerPlatformRoleName':                         'category',
            'Census_InternalBatteryType':                           'category',
            'Census_InternalBatteryNumberOfCharges':                'float32',
            'Census_OSVersion':                                     'category',
            'Census_OSArchitecture':                                'category',
            'Census_OSBranch':                                      'category',
            'Census_OSBuildNumber':                                 'int32',
            'Census_OSBuildRevision':                               'int32',
            'Census_OSEdition':                                     'category',
            'Census_OSSkuName':                                     'category',
            'Census_OSInstallTypeName':                             'category',
            'Census_OSInstallLanguageIdentifier':                   'float32',
            'Census_OSUILocaleIdentifier':                          'int32',
            'Census_OSWUAutoUpdateOptionsName':                     'category',
            'Census_IsPortableOperatingSystem':                     'Int8',
            'Census_GenuineStateName':                              'category',
            'Census_ActivationChannel':                             'category',
            'Census_IsFlightingInternal':                           'float32',
            'Census_IsFlightsDisabled':                             'float32',
            'Census_FlightRing':                                    'category',
            'Census_ThresholdOptIn':                                'float32',
            'Census_FirmwareManufacturerIdentifier':                'float32',
            'Census_FirmwareVersionIdentifier':                     'float32',
            'Census_IsSecureBootEnabled':                           'Int8',
            'Census_IsWIMBootEnabled':                              'float32',
            'Census_IsVirtualDevice':                               'float32',
            'Census_IsTouchEnabled':                                'Int8',
            'Census_IsPenCapable':                                  'Int8',
            'Census_IsAlwaysOnAlwaysConnectedCapable':              'float32',
            'Wdft_IsGamer':                                         'float32',
            'Wdft_RegionIdentifier':                                'float32',
            'HasDetections':                                        'Int8'
        }

    print("loading training data")
    train_df = pd.read_csv('train.csv', dtype=dtypes, nrows=1000000)
    train_df.drop(columns=INDEX, axis=1, inplace=True)                      # ignore machine identifier
    has_detections = train_df.pop(LABELS)

    # Fill missing
    print("filling missing entries")
    fill_missing(train_df)
    display_missing(train_df)

    # Label encode categorical variables
    print("label encoding categorical features")
    categorical = [col for col in train_df.columns if train_df[col].dtype == 'category']
    for col in categorical:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))

    # FEATURE SELECTION
    # Separate features and target
    X = train_df
    y = has_detections

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # display t-SNE
    display_tSNE(X_scaled)

    # PCA
    pca = PCA(n_components=0.95)
    X_low = pca.fit_transform(X_scaled)
    # LDA
    # lda = LinearDiscriminantAnalysis(n_components=1)
    # lda.fit(X_scaled, y)
    # X_low = lda.transform(X_scaled)

    print(f"Original training data shape: {X_scaled.shape}")
    print(f"LDA transformed training data shape: {X_low.shape}")
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train_lda, X_val_lda, y_train, y_val = train_test_split(X_low, y, test_size=0.2, random_state=42)

    from sklearn.tree import DecisionTreeClassifier

    # LightGBM model for standard scaled data
    lgb_model_scaled = lgb.LGBMClassifier(random_state=42)
    lgb_model_scaled.fit(X_train_scaled, y_train)

    # Make predictions on validation set
    y_pred_scaled = lgb_model_scaled.predict_proba(X_val_scaled)[:, 1]
    auc_scaled = roc_auc_score(y_val, y_pred_scaled)

    print(f"---- AUC score for Standard Scaled data: {auc_scaled} ----")

    # LightGBM model for LDA transformed data
    lgb_model_lda = lgb.LGBMClassifier(random_state=42)
    lgb_model_lda.fit(X_train_lda, y_train)

    # Make predictions on validation set
    y_pred_lda = lgb_model_lda.predict_proba(X_val_lda)[:, 1]
    auc_lda = roc_auc_score(y_val, y_pred_lda)

    print(f"---- AUC score for LDA transformed data: {auc_lda} ----")

if __name__ == '__main__':
    main()