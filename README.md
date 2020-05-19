# analysis_for_kaggle
### overview
- single model
  - regression
  - classification(multilabel/multiclass)
- ensemble model

### available model
##### classification
- log_reg
  - LogisticRegression
- log_reg_cv
  - LogisticRegressionCV
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
  - SGDClassifier
- dt_clf
  - DecisionTreeClassifier
- xgb_clf
  - XGBClassifier
- lgb_clf
  - LGBMClassifier
- catb_clf
  - CatBoostClassifier
- rgf_clf
  - RGFClassifier
- keras_clf
  - KerasClassifier
- torch_clf
  - NeuralNetClassifier
- bert_clf
  - BertClassifier

##### regression
- linear_reg
  - LinearRegression
- lasso
  - Lasso
- ridge
  - Ridge
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
- catb_reg
  - CatBoostRegressor
- rgf_reg
  - RGFRegressor
- keras_reg
  - KerasRegressor
- torch_reg
  - NeuralNetRegressor
- bert_reg
  - BertRegressor

### CUI flow
- cp analysis_for_kaggle/configs/config.json.org configs/config.json
- **update config.json**
- python analysis_for_kaggle/predict.py

### environment
- https://github.com/hidetomo-watanabe/analysis_dockerfiles
