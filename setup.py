import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as roc_auc, accuracy_score as acc
from sklearn.ensemble import RandomForestClassifier as RFC

seed = 42

import warnings
warnings.filterwarnings("ignore")

def rfc(train_X, train_Y, validation_X, validation_Y, seed=42, class_weight={0:1, 1:1}):

    rfc = RFC(n_estimators=100, random_state=seed, class_weight=class_weight)
    rfc = rfc.fit(train_X, train_Y)

    print("ROC-AUC: {0:.2f}".format(roc_auc(validation_Y, rfc.predict(validation_X))))

def wrangle_data(train):

    train = train.copy()
    train.columns = train.columns.str.replace('-','_')
    train.fillna(train.mean().astype(int), inplace=True)

    train['n_unpaid'] = train.Count_3_6_months_late +train.Count_6_12_months_late + train.Count_more_than_12_months_late

    train = pd.merge(
        train[train.columns[~train.columns.isin(train.select_dtypes(object))]],
        pd.get_dummies(train.select_dtypes(object)),
        left_index=True, right_index=True
    )

    return train

def split_data(train, scaled=True, segmented=False):

    # split the dataset once again
    train_X, validation_X, train_Y, validation_Y = train_test_split(train.drop(['renewal'], axis=1),
                                                   train['renewal'],
                                                   train_size = .8,
                                                   random_state = seed
                                                   )
    if segmented:
        # determine the feature range for the predictor variables
        segment = train_X.loc[list(train_Y[train_Y==0].index),:].describe()

        # segment the training set based on the max-min limits
        for column in segment.columns:
            train_X = train_X.loc[(train_X[column]<=segment.loc['max', column]) & (train_X[column]>=segment.loc['min', column])]

        train_Y = train_Y.loc[train_X.index]

    if scaled:
        #scale data
        scaler = MinMaxScaler()

        train_Y = train_Y.values.reshape((len(train_Y),))
        validation_Y = validation_Y.values.reshape((len(validation_Y),))

        # scale values to range 0-1
        train_X = scaler.fit_transform(train_X)
        validation_X = scaler.transform(validation_X)

    return train_X, validation_X, train_Y, validation_Y
