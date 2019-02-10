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
flake8_msg=${flake8_msg}$(flake8 modules/ --ignore E402 | grep -v \'myfunc\')
flake8_msg=${flake8_msg}$(flake8 for_kernel/ --ignore E402 --ignore F821 | grep -v predict_for_kernel.py)
