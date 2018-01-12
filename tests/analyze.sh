#!/bin/bash
rm outputs/tmp.csv
python -u analyze.py tests/titanic/test_config.json > /dev/null
diff_msg=$(diff outputs/tmp.csv tests/titanic/output.csv)
if [ -n "${diff_msg}" ];then
  echo 'INTEGRATION TEST ERROR'
  echo -e "${diff_msg}"
else
  echo 'INTEGRATION TEST OK'
fi
