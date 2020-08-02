from sklearn import model_selection
import pandas as pd

class CrossValidation:
    def __init__(self, df, target, problem_type = 'binary_classification', num_folds = 5, shuffle = True, random_state = 100, multilabel_delimiter = ','):
        self.df = df
        self.target = target
        self.target_len = len(target)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.multilabel_delimiter = multilabel_delimiter
        self.random_state = random_state

        if self.shuffle is True:
            self.df = self.df.sample(frac = 1).reset_index(drop= True)
        self.df['kfold'] = -1
    
    def split(self):
        if self.problem_type in ['binary_classification', 'multiclass_clssification']:
            unique_value = self.df[self.target[0]].nunique()
            if unique_value == 1:
                raise Exception('Only one unique value found')
            elif unique_value > 1:
                target = self.target[0]
                kf = model_selection.StratifiedKFold(n_splits = self.num_folds, shuffle = False, random_state = self.random_state)
                for fold,(train_data, val_data) in enumerate(kf.split(X = self.df, y = self.df.target.values)):
                    print(len(train_data), len(val_data))
                    self.df.loc[val_data, 'kfold'] = fold
        elif self.problem_type in ['singlecol_regression', 'multicol_regression']:
            if self.target_len != 1 and self.problem_type == 'singlecol_regression':
                raise Exception('invalid no of targets')
            if self.target_len < 2 and self.problem_type == 'multicol_regression':
                raise Exception('Invalid number of targets')
            kf = model_selection.KFold(n_splits = self.num_folds, shuffle = False)
            for fold,(train_data, val_data) in enumerate(kf.split(X = self.df)):
                    print(len(train_data), len(val_data))
                    self.df.loc[val_data, 'kfold'] = fold
        
        elif self.problem_type.startswith('holdout_'):
            holdout_percentage = int(self.problem_type.split('_')[1])
            holdout_sample = int(len(self.df) * holdout_percentage/100)
            self.df.loc[:len(self.df)-holdout_sample, 'kfold'] = 0
            self.df.loc[len(self.df)-holdout_sample:,'kfold'] = 1

        elif self.problem_type == 'multilabel_classification':
            if self.target_len != 1:
                raise Exception('invalid number of targets')
            targets = self.df[self.target[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits = self.num_folds, shuffle = False, random_state = self.random_state)
            for fold,(train_data, val_data) in enumerate(kf.split(X = self.df, y = targets)):
                print(len(train_data), len(val_data))
                self.df.loc[val_data, 'kfold'] = fold
        
        else:
            raise Exception('Problem type not understood')


        return self.df

if __name__ == '__main__':
    df = pd.read_csv('input/train.csv')
    cv = CrossValidation(df, target = ['target'], problem_type = 'holdout_10')
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
