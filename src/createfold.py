import pandas as pd
from sklearn import model_selection
if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    df['kfold'] = -1

    df = df.sample(frac = 1).reset_index(drop = True)
    kf = model_selection.StratifiedKFold(n_splits = 5, shuffle = False, random_state = 100)
    for fold,(train_data, val_data) in enumerate(kf.split(X = df, y = df.target.values)):
        print(len(train_data), len(val_data))
        df.loc[val_data, 'kfold'] = fold
        df.to_csv('input/train_folds.csv', index = False)