from logging import Formatter, INFO, StreamHandler, getLogger

if __name__ == '__main__':
    logger = getLogger('predict')
    handler = StreamHandler()
    formatter = Formatter(
        '[%(asctime)s][%(levelname)s](%(filename)s:%(lineno)s) %(message)s')
    handler.setLevel(INFO)
    handler.setFormatter(formatter)
    logger.setLevel(INFO)
    logger.addHandler(handler)
    logger.propagate = False

    logger.info('# START')

    # translate
    translater_obj = DataTranslater(kernel=True)
    translater_obj.read_config_text(config_text)
    translater_obj.get_translater()

    logger.info('### TRANSLATE')
    translater_obj.calc_train_data()

    logger.info('### WRITE TRAIN DATA')
    translater_obj.write_train_data()

    logger.info('### GET TRAIN DATA')
    train_data = translater_obj.get_train_data()
    post_processers = translater_obj.get_post_processers()

    # train
    trainer_obj = Trainer(**train_data, kernel=True)
    trainer_obj.read_config_text(config_text)

    logger.info('### FIT')
    trainer_obj.calc_estimator_data()

    logger.info('### WRITE ESTIMATOR DATA')
    trainer_obj.write_estimator_data()

    logger.info('### GET ESTIMATOR DATA')
    estimator_data = trainer_obj.get_estimator_data()

    # output
    outputer_obj = Outputer(
        **train_data, **estimator_data, **post_processers, kernel=True)
    outputer_obj.read_config_text(config_text)

    logger.info('### PREDICT')
    outputer_obj.calc_predict_data()

    logger.info('### WRITE PREDICT DATA')
    outputer_obj.write_predict_data()

    logger.info('# FINISHED')
