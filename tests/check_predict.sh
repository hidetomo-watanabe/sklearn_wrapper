#!/bin/bash

set -eu
trap catch EXIT
catch()
{
  if [ $? -ne 0 ]; then
    echo 'PREDICT TEST ERROR'
  fi
}

if [ -e 'outputs/tmp.csv' ]; then
  rm outputs/tmp.csv
fi

# titanic
err_msg=$(python -u scripts/predict.py tests/titanic/test_config.json | grep ERROR | :)
err_msg=${err_msg}$(python -u scripts/predict.py tests/house/test_config.json | grep ERROR | :)
if [ -n "${err_msg}" ];then
  echo 'PREDICT TEST ERROR'
  echo -e "${err_msg}"
  exit 1
fi

diff_msg=$(diff outputs/tmp_titanic.csv tests/titanic/output.csv)
diff_msg=${diff_msg}$(diff outputs/tmp_house.csv tests/house/output.csv)
if [ -n "${diff_msg}" ];then
  echo 'PREDICT TEST ERROR'
  echo -e "${diff_msg}"
  exit 1
fi

diff_msg=$(diff outputs/proba_tmp_titanic.csv tests/titanic/proba_output.csv)
if [ -n "${diff_msg}" ];then
  echo 'PREDICT TEST ERROR'
  echo -e "${diff_msg}"
  exit 1
fi

echo 'PREDICT TEST OK'
