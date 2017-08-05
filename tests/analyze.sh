#!/bin/bash
python -u analyze.py tests/titanic/test_config.ini > /dev/null
diff_msg=$(diff outputs/tmp.csv tests/titanic/output.csv)
if [ -n "${diff_msg}" ];then
  echo 'INTEGRATION TEST ERROR'
  echo -e "${diff_msg}"
else
  echo 'INTEGRATION TEST OK'
fi
