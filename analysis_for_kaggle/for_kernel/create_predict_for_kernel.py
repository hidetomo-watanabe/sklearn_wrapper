import sys
import os
import json


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
        config_path = '%s/configs/config.json' % BASE_PATH

    FILENAME = '%s/for_kernel/predict_for_kernel.py' % BASE_PATH
    # config
    with open(FILENAME, 'w') as f:
        f.write('config_text = """\\\n')
    _append_file_text(config_path, FILENAME)
    with open(FILENAME, 'a') as f:
        f.write('"""\n')
    # modules
    _append_file_text(
        '%s/modules/ConfigReader.py' % BASE_PATH, FILENAME)
    _append_file_text(
        '%s/modules/CommonDataTranslater.py' % BASE_PATH, FILENAME)
    _append_file_text(
        '%s/modules/TableDataTranslater.py' % BASE_PATH, FILENAME)
    _append_file_text(
        '%s/modules/ImageDataTranslater.py' % BASE_PATH, FILENAME)
    _append_file_text(
        '%s/modules/DataTranslater.py' % BASE_PATH, FILENAME)
    _append_file_text(
        '%s/modules/Trainer.py' % BASE_PATH, FILENAME)
    _append_file_text(
        '%s/modules/Outputer.py' % BASE_PATH, FILENAME)
    # adhoc
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())
    adhocs = []
    if config_json['data']['type'] == 'table':
        if config_json['pre']['table'].get('adhoc_df'):
            adhocs.append(config_json['pre']['table']['adhoc_df']['myfunc'])
    if config_json['fit'].get('myfunc'):
        adhocs.append(config_json['fit']['myfunc'])
    if config_json['post'].get('myfunc'):
        adhocs.append(config_json['post']['myfunc'])
    adhocs = list(set(adhocs))
    for adhoc in adhocs:
        _append_file_text(
            '%s/modules/myfuncs/%s.py' % (BASE_PATH, adhoc), FILENAME)
    # base
    _append_file_text(
        '%s/for_kernel/predict_for_kernel_base.py' % BASE_PATH, FILENAME)
