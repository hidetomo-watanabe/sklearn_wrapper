#!/bin/bash
echo '# CHECK WITH FLAKE8'
sh tests/check_with_flake8.sh
echo '# CHECK PREDICT'
sh tests/check_predict.sh
