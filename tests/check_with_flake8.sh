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

flake8_msg=$(flake8 analysis_for_kaggle/ --ignore E402,F821,F841)
