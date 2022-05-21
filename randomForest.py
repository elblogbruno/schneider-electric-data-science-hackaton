from sklearn.ensemble import RandomForestClassifier


sz = len(data.columns)
targetName = "targetRelease"

colnames = data.columns.values.tolist()
colnames.remove(targetName)

predictors = colnames
target = targetName

forest = RandomForestClassifier(n_jobs=1, oob_score=True, n_estimators=10)
forest.fit(data[predictors], data[target])

#ver resultados de la funcion de decision
forest.oob_decision_function_
forest.oob_score_
