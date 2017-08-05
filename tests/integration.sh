#!/bin/bash
cp config.ini config.ini_bk
cp configs/titanic.ini config.ini
python -u analyze.py > /dev/null
mv config.ini_bk config.ini
diff outputs/tmp.csv tests/titanic_output.csv
