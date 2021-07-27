# sklearn_wrapper
### overview
- single/ensemble model
- classification/regression
- table/image

### data flow
##### pre processing
- feature engineering
- validation(adversarial, no anomaly)

##### pipeline
- image augmentation
- sampling(under, over)

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
- tabnet_clf
  - TabNetClassifier

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
- tabnet_reg
  - TabNetRegressor

### CUI flow
- cp sklearn_wrapper/configs/config.json.org sklearn_wrapper/configs/config.json
- **update config.json**
- python sklearn_wrapper/predict.py

### environment
- https://github.com/hidetomo-watanabe/analysis_dockerfiles
