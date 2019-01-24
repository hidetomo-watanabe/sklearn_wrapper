#!/bin/bash

set -eu
trap catch EXIT
catch()
{
  if [ $? -ne 0 ]; then
    echo 'PREDICT TEST ERROR: SHELL'
    echo -e "${err_msg}"
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

echo 'START PREDICT'
# titanic
# gbdt
err_msg=$(python -u predict.py tests/titanic/test_config.json | grep ERROR | :)
# lgb
err_msg=${err_msg}$(python -u predict.py tests/titanic/test_config2.json | grep ERROR | :)
# house
# svr
err_msg=${err_msg}$(python -u predict.py tests/house/test_config.json | grep ERROR | :)
if [ -n "${err_msg}" ];then
  echo 'PREDICT TEST ERROR: PREDICT'
  echo -e "${err_msg}"
  exit 1
fi

echo 'START DIFF'
# diff
diff_msg=$(diff outputs/tmp_titanic.csv tests/titanic/output.csv)
diff_msg=${diff_msg}$(diff outputs/tmp_titanic2.csv tests/titanic/output2.csv)
diff_msg=${diff_msg}$(diff outputs/tmp_house.csv tests/house/output.csv)
if [ -n "${diff_msg}" ];then
  echo 'PREDICT TEST ERROR: DIFF'
  echo -e "${diff_msg}"
  exit 1
fi

echo 'START DIFF PROBA'
# diff proba
diff_msg=$(diff outputs/proba_tmp_titanic.csv tests/titanic/proba_output.csv)
diff_msg=${diff_msg}$(diff outputs/proba_tmp_titanic2.csv tests/titanic/proba_output2.csv)
if [ -n "${diff_msg}" ];then
  echo 'PREDICT TEST ERROR: DIFF PROBA'
  echo -e "${diff_msg}"
  exit 1
fi

echo 'PREDICT TEST OK'
