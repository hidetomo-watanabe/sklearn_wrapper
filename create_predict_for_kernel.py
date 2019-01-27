import sys
import json


def _append_file_text(before, after):
    with open(before, 'r') as f:
        text = f.read()
    with open(after, 'a') as f:
        f.write(text)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = './configs/config.json'

    FILENAME = './predict_for_kernel.py'
    # config
    with open(FILENAME, 'w') as f:
        f.write('config_text = """\\\n')
    _append_file_text(config_path, FILENAME)
    with open(FILENAME, 'a') as f:
        f.write('"""\n')
    # modules
    _append_file_text('./modules/Predicter.py', FILENAME)
    _append_file_text('./modules/Notifier.py', FILENAME)
    _append_file_text('./modules/MyLogger.py', FILENAME)
    # translate adhoc
    with open(config_path, 'r') as f:
        config_json = json.loads(f.read())
    adhoc = config_json['translate']['adhoc']['myfunc']
    if adhoc:
        _append_file_text('./modules/myfuncs/%s.py' % adhoc, FILENAME)
    # keras model
    single_models = config_json['fit']['single_models']
    for model in single_models:
        keras_build = model.get('keras_build')
        if keras_build:
            _append_file_text(
                './modules/mykerasmodels/%s.py' % keras_build, FILENAME)
    # base
    _append_file_text('./predict_for_kernel_base.py', FILENAME)
