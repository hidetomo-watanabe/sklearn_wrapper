#!/bin/bash

# set -eu
# trap _check EXIT
_check()
{
  if [ -n "${err_msg}" ]; then
    echo "${err_msg}"
    echo 'INTEGRATE TEST ERROR'
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
_del 'outputs/tmp_titanic5.csv'

err_msg=''
echo 'START INTEGRATE'
# titanic table binary classification
echo '  TITANIC'
# vote
err_msg=${err_msg}$(python -u integrate.py tests/titanic/test_config5.json 2>&1 | grep ERROR)
_check

# diff
echo 'START DIFF'
echo '  TITANIC'
err_msg=${err_msg}$(diff outputs/tmp_titanic5.csv tests/titanic/output5.csv)

# finish
echo 'INTEGRATE TEST OK'
