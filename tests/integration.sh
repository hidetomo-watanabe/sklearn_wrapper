#!/bin/bash
python -u analyze.py configs/titanic.ini > /dev/null
diff_msg=$(diff outputs/tmp.csv tests/titanic_output.csv)
if [ -n "${diff_msg}" ];then
  echo 'INTEGRATION TEST ERROR'
  echo -e "${diff_msg}"
else
  echo 'INTEGRATION TEST OK'
fi
