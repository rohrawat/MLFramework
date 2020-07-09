import os
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

Test_Data = 'input/test.csv'
Model = input()

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,3]
}
FOLD = 0
def predict():
    df = pd.read_csv(Test_Data)
    test_id = df['id'].values
    prediction = None
    
    #for FOLD in range(5):
    df = pd.read_csv(Test_Data)
    label_encoder = joblib.load(os.path.join('models',f'{Model}_{FOLD}_label_encoder.pkl'))
    column = joblib.load(os.path.join('models',f'{Model}_{FOLD}_column.pkl'))
    for col in column:
        le = label_encoder[col]
        df.loc[:,col] = le.transform(df[col].values.tolist())

    clf = joblib.load(os.path.join('models',f'{Model}_{FOLD}.pkl'))
                                                                
    df = df[column]
    ypred = clf.predict_proba(df)[:,1]
    if FOLD == 0:
        prediction = ypred
    else:
        prediction += ypred
    #prediction /= 5
    submit = pd.DataFrame(np.column_stack((test_id, prediction)), columns = ['id', 'target'])
    return submit

if __name__ == "__main__":
    submit = predict()
    submit.id = submit.id.astype(int)
    submit.to_csv(f'models/{Model}.csv', index = False)