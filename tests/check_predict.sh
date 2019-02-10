#!/bin/bash

set -eu
trap catch EXIT
catch()
{
  if [ -n "${err_msg}" ]; then
    echo 'PREDICT TEST ERROR'
    echo "${err_msg}"
    exit 1
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

echo 'START PREDICT'
# titanic
# gbdt
python -u predict.py tests/titanic/test_config.json
# lgb
python -u predict.py tests/titanic/test_config2.json
# keras
python -u predict.py tests/titanic/test_config3.json
# house
# svr
python -u predict.py tests/house/test_config.json
# keras
python -u predict.py tests/house/test_config2.json

err_msg=''
echo 'START DIFF'
# diff
err_msg=${err_msg}$(diff outputs/tmp_titanic.csv tests/titanic/output.csv)
err_msg=${err_msg}$(diff outputs/tmp_titanic2.csv tests/titanic/output2.csv)
err_msg=${err_msg}$(diff outputs/tmp_house.csv tests/house/output.csv)

echo 'START DIFF PROBA'
# diff proba
err_msg=${err_msg}$(diff outputs/proba_tmp_titanic.csv tests/titanic/proba_output.csv)
err_msg=${err_msg}$(diff outputs/proba_tmp_titanic2.csv tests/titanic/proba_output2.csv)

echo 'PREDICT TEST OK'
