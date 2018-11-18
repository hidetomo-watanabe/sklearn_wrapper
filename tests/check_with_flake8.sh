#!/bin/bash
flake8_msg=$(flake8 modules/ --ignore E402 | grep -v myfunc)
flake8_msg=${flake8_msg}$(flake8 myfuncs/ --ignore E402)
flake8_msg=${flake8_msg}$(flake8 scripts/ --ignore E402)
if [ -n "${flake8_msg}" ];then
  echo 'FLAKE8 TEST ERROR'
  echo -e "${flake8_msg}"
else
  echo 'FLAKE8 TEST OK'
fi
