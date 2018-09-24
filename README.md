# analysis_for_kaggle
### overview
- only SINGLE MODEL analysis
- (TBI) ensemble model

### available model
##### classification
- log_reg
  - LogisticRegression
- svc
  - SVC
- l_svc
  - LinearSVC
- rf_clf
  - RandomForestClassifier
- gbdt_clf
  - GradientBoostingForestClassifier
- knn_clf
  - KNeighborsClassifier
- g_nb
  - GaussianNB
- perceptron
  - Perceptron
- sgd_clf
  - SGDClasssifier
- dt_clf
  - DecisionTreeClasssifier
- xgb_clf
  - XGBClasssifier
- lgb_clf
  - LGBMClasssifier

##### regression
- linear_reg
  - LinearRegression
- svr
  - SVR
- l_svr
  - LinearSVR
- rf_reg
  - RandomForestRegressor
- gbdt_reg
  - GradientBoostingForestRegressor
- knn_reg
  - KNeighborsRegressor
- sgd_reg
  - SGDRegressor
- dt_reg
  - DecisionTreeRegressor
- xgb_reg
  - XGBRegressor
- lgb_reg
  - LGBMRegressor

### sample configs
- titanic
- digit
- house

### CUI flow
- cp scripts/config.json.org scripts/config.json
- (update config.json)
- python -u scripts/analyze.py
