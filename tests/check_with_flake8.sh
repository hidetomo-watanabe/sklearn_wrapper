#!/bin/bash

# set -eu
trap catch EXIT
catch()
{
  if [ -n "${flake8_msg}" ]; then
    echo "${flake8_msg}"
    echo 'FLAKE8 TEST ERROR'
    exit 1
  else
    echo 'FLAKE8 TEST OK'
  fi
}

flake8_msg=''
flake8_msg=${flake8_msg}$(flake8 analysis_for_kaggle/*.py --ignore E402)
flake8_msg=${flake8_msg}$(flake8 analysis_for_kaggle/modules/ --ignore E402 | grep -v \'myfunc\')
flake8_msg=${flake8_msg}$(flake8 analysis_for_kaggle/for_kernel/ --ignore E402 --ignore F821 | grep -v predict_for_kernel.py)
