#!/bin/bash

# set -eu
trap catch EXIT
catch()
{
  if [ -n "${err_msg}" ]; then
    echo "${err_msg}"
    echo 'PREDICT TEST ERROR'
    exit 1
  else
    echo 'PREDICT TEST OK'
  fi
}

_del()
{
  if [ -e $1 ]; then
    rm $1
  fi
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
# titanic binary classification
# gbdt
err_msg=${err_msg}$(python -u predict.py tests/titanic/test_config.json 2>&1 | grep ERROR)
# lgb
err_msg=${err_msg}$(python -u predict.py tests/titanic/test_config2.json 2>&1 | grep ERROR)
# keras
err_msg=${err_msg}$(python -u predict.py tests/titanic/test_config3.json 2>&1 | grep ERROR)
# ensemble
err_msg=${err_msg}$(python -u predict.py tests/titanic/test_config4.json 2>&1 | grep ERROR)
# house regression
# svr
err_msg=${err_msg}$(python -u predict.py tests/house/test_config.json 2>&1 | grep ERROR)
# keras
err_msg=${err_msg}$(python -u predict.py tests/house/test_config2.json 2>&1 | grep ERROR)
# digit part multi lable classification
# lgb
err_msg=${err_msg}$(python -u predict.py tests/digit_part/test_config.json 2>&1 | grep ERROR)

echo 'START DIFF'
# diff
err_msg=${err_msg}$(diff outputs/tmp_titanic.csv tests/titanic/output.csv)
err_msg=${err_msg}$(diff outputs/tmp_titanic2.csv tests/titanic/output2.csv)
err_msg=${err_msg}$(diff outputs/tmp_titanic4.csv tests/titanic/output4.csv)
err_msg=${err_msg}$(diff outputs/tmp_house.csv tests/house/output.csv)

echo 'START DIFF PROBA'
# diff proba
err_msg=${err_msg}$(diff outputs/proba_tmp_titanic.csv tests/titanic/proba_output.csv)
err_msg=${err_msg}$(diff outputs/proba_tmp_titanic2.csv tests/titanic/proba_output2.csv)
