import json
import os
import sys


def _append_file_text(before, after):
    with open(before, 'r') as f:
        text = f.read()
    with open(after, 'a') as f:
        f.write(text)


if __name__ == '__main__':
    BASE_PATH = '%s/..' % os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = f'{BASE_PATH}/configs/config.json'

    FILENAME = f'{BASE_PATH}/for_kernel/predict_for_kernel.py'
    # config
    with open(FILENAME, 'w') as f:
        f.write('config_text = """\\\n')
    _append_file_text(config_path, FILENAME)
    with open(FILENAME, 'a') as f:
        f.write('"""\n')
    # modules
    # keep append order
    _append_file_text(
        f'{BASE_PATH}/modules/ConfigReader.py', FILENAME)
    _append_file_text(
        f'{BASE_PATH}/modules/CommonMethodWrapper.py', FILENAME)
    _append_file_text(
        f'{BASE_PATH}/modules/Outputer.py', FILENAME)
    for prefix in ['Base', 'Ensemble', 'Single', '']:
        _append_file_text(
            f'{BASE_PATH}/modules/trainers/{prefix}Trainer.py', FILENAME)
    for prefix in ['Base', 'Table', 'Image', '']:
        _append_file_text(
            f'{BASE_PATH}/modules/data_translaters/{prefix}DataTranslater.py',
            FILENAME)
    # adhoc
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())
    adhocs = []
    if config_json['data']['type'] == 'table':
        if config_json['pre']['table'].get('adhoc_df'):
            adhocs.append(config_json['pre']['table']['adhoc_df']['myfunc'])
    if config_json['fit'].get('myfunc'):
        adhocs.append(config_json['fit']['myfunc'])
    if config_json['post']:
        adhocs.append(config_json['post']['myfunc'])
    adhocs = list(set(adhocs))
    for adhoc in adhocs:
        _append_file_text(
            f'{BASE_PATH}/modules/myfuncs/{adhoc}.py', FILENAME)
    # base
    _append_file_text(
        f'{BASE_PATH}/for_kernel/predict_for_kernel_base.py', FILENAME)
