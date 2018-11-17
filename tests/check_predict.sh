#!/bin/bash
if [ -e 'outputs/tmp.csv' ]; then
  rm outputs/tmp.csv
fi

err_msg=$(python -u scripts/predict.py tests/titanic/test_config.json | grep ERROR)
if [ -n "${err_msg}" ];then
  echo 'INTEGRATION TEST ERROR'
  echo -e "${err_msg}"
  exit 1
fi

diff_msg=$(diff outputs/tmp.csv tests/titanic/output.csv)
if [ -n "${diff_msg}" ];then
  echo 'INTEGRATION TEST ERROR'
  echo -e "${diff_msg}"
  exit 1
fi

diff_msg=$(diff outputs/proba_tmp.csv tests/titanic/proba_output.csv)
if [ -n "${diff_msg}" ];then
  echo 'INTEGRATION TEST ERROR'
  echo -e "${diff_msg}"
  exit 1
fi

echo 'INTEGRATION TEST OK'
