import os
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib

from . import dispatcher

Training_Data = 'input/train_folds.csv'
Test_Data = 'input/test.csv'
FOLD = [0,1,2,3,4]
Model = input()

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}
if __name__ == "__main__":
    df = pd.read_csv(Training_Data)
    test_data = pd.read_csv(Test_Data)
    for fold in FOLD:
        train_data = df[df.kfold.isin(FOLD_MAPPING.get(fold))]
        val_data = df[df.kfold == fold]

        y_train = train_data.target.values
        y_val = val_data.target.values

        train_data = train_data.drop(['id','target','kfold'], axis = 1)
        val_data = val_data.drop(['id', 'target', 'kfold'], axis = 1)

        val_data = val_data[train_data.columns]

        label_encoders = {}

        for col in train_data.columns:
            le = preprocessing.LabelEncoder()
            le.fit(train_data[col].values.tolist()+val_data[col].values.tolist()+test_data[col].values.tolist())
            train_data.loc[:,col] = le.transform(train_data[col].values.tolist())
            val_data.loc[:,col] = le.transform(val_data[col].values.tolist())
            label_encoders[col]= le

        clf = dispatcher.Models[Model]
        clf.fit(train_data, y_train)
        ypred = clf.predict_proba(val_data)[:,1]
        print(metrics.roc_auc_score(y_val,ypred))

        joblib.dump(label_encoders, f"models/{Model}_{fold}_label_encoder.pkl")
        joblib.dump(clf, f"models/{Model}_{fold}.pkl")
        joblib.dump(train_data.columns,f'Models/{Model}_{fold}_column.pkl')