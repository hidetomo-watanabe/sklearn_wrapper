#!/bin/bash
echo '# CHECK WITH FLAKE8'
sh tests/check_with_flake8.sh
echo '# CHECK PREDICT'
python tests/test_predict.py
