#!/bin/bash

CONFIG_FILE=$1
echo 'config_text = """\' > predict_for_kernel.py
cat ${CONFIG_FILE} >> predict_for_kernel.py
echo '"""' >> predict_for_kernel.py

cat modules/Predicter.py >> predict_for_kernel.py
cat modules/Notifier.py >> predict_for_kernel.py
cat modules/MyLogger.py >> predict_for_kernel.py
cat predict_for_kernel_base.py >> predict_for_kernel.py
