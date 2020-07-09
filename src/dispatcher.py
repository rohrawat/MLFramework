from sklearn import ensemble

Models ={
    "randomforrest": ensemble.RandomForestClassifier(n_estimators = 100, n_jobs = -1, verbose = 2),
    "extratreeclssifier":ensemble.ExtraTreesClassifier(n_estimators = 200,n_jobs = -1, verbose = 2)
}