#!/bin/bash
flake8_msg=$(flake8 *.py --ignore F401)
flake8_msg=${flake8_msg}$(flake8 myfuncs/*.py --ignore F401)
if [ -n "${flake8_msg}" ];then
  echo 'FLAKE8 TEST ERROR'
  echo -e "${flake8_msg}"
else
  echo 'FLAKE8 TEST OK'
fi
