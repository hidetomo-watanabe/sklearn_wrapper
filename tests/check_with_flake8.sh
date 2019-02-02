#!/bin/bash

set -eu
trap catch EXIT
catch()
{
  if [ -n "${flake8_msg}" ]; then
    echo 'FLAKE8 TEST ERROR'
    echo -e "${flake8_msg}"
  fi
}

flake8_msg=''
flake8_msg=${flake8_msg}$(flake8 modules/ --ignore E402 | grep -v myfunc)
flake8_msg=${flake8_msg}$(flake8 for_kernel/ --ignore E402)

echo 'FLAKE8 TEST OK'
