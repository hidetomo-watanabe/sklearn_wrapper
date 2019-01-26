import sys


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
    with open(FILENAME, 'w') as f:
        f.write('config_text = """\\\n')
    _append_file_text(config_path, FILENAME)
    with open(FILENAME, 'a') as f:
        f.write('"""\n')
    _append_file_text('./modules/Predicter.py', FILENAME)
    _append_file_text('./modules/Notifier.py', FILENAME)
    _append_file_text('./modules/MyLogger.py', FILENAME)
    _append_file_text('./predict_for_kernel_base.py', FILENAME)
