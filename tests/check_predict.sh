#!/bin/bash

# set -eu
# trap _check EXIT
_check()
{
  if [ -n "${err_msg}" ]; then
    echo "${err_msg}"
    echo 'PREDICT TEST ERROR'
    exit 1
  fi
  return
}

_del()
{
  if [ -e $1 ]; then
    rm $1
  fi
  return
}

# del old csv
echo 'DEL OLD CSV'
_del 'outputs/tmp_titanic.csv'
_del 'outputs/tmp_titanic2.csv'
_del 'outputs/proba_tmp_titanic.csv'
_del 'outputs/proba_tmp_titanic2.csv'
_del 'outputs/tmp_house.csv'
_del 'outputs/tmp_house2.csv'

err_msg=''
echo 'START PREDICT'
# titanic table binary classification
echo '  TITANIC'
# gbdt
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/titanic/test_config.json 2>&1 | grep ERROR)
# lgb
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/titanic/test_config2.json 2>&1 | grep ERROR)
# keras
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/titanic/test_config3.json 2>&1 | grep ERROR)
# ensemble(stacking)
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/titanic/test_config4.json 2>&1 | grep ERROR)
# ensemble(vote)
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/titanic/test_config5.json 2>&1 | grep ERROR)
# pseudo
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/titanic/test_config6.json 2>&1 | grep ERROR)
_check
# house table regression
echo '  HOUSE'
# svr
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/house/test_config.json 2>&1 | grep ERROR)
# keras
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/house/test_config2.json 2>&1 | grep ERROR)
_check
# digit_part table multi lable classification
echo '  DIGIT PART'
# lgb
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/digit_part/test_config.json 2>&1 | grep ERROR)
# keras(lstm)
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/digit_part/test_config2.json 2>&1 | grep ERROR)
_check
# cactus_part image binary classification
echo '  CACTUS PART'
# keras(vgg16)
err_msg=${err_msg}$(python -u analysis_for_kaggle/predict.py tests/cactus_part/test_config.json 2>&1 | grep ERROR)
_check

# diff
echo 'START DIFF'
echo '  TITANIC'
err_msg=${err_msg}$(diff outputs/tmp_titanic.csv tests/titanic/output.csv)
err_msg=${err_msg}$(diff outputs/tmp_titanic2.csv tests/titanic/output2.csv)
err_msg=${err_msg}$(diff outputs/tmp_titanic4.csv tests/titanic/output4.csv)
err_msg=${err_msg}$(diff outputs/tmp_titanic5.csv tests/titanic/output5.csv)
err_msg=${err_msg}$(diff outputs/tmp_titanic6.csv tests/titanic/output6.csv)
echo '  HOUSE'
err_msg=${err_msg}$(diff outputs/tmp_house.csv tests/house/output.csv)
_check

# diff proba
echo 'START DIFF PROBA'
echo '  TITANIC'
err_msg=${err_msg}$(diff outputs/proba_tmp_titanic.csv tests/titanic/proba_output.csv)
err_msg=${err_msg}$(diff outputs/proba_tmp_titanic2.csv tests/titanic/proba_output2.csv)
_check

# finish
echo 'PREDICT TEST OK'
