from logging import getLogger, StreamHandler, Formatter, DEBUG

if __name__ == '__main__':
    logger = getLogger('predict')
    handler = StreamHandler()
    formatter = Formatter(
        '[%(asctime)s][%(levelname)s](%(filename)s:%(lineno)s) %(message)s')
    handler.setLevel(DEBUG)
    handler.setFormatter(formatter)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    logger.info('# START')

    # data translate
    translater_obj = DataTranslater(kernel=True)
    translater_obj.read_config_text(config_text)
    translater_obj.get_translater()

    logger.info('### DATA FOR VIEW')
    translater_obj.create_data_for_view()
    translater_obj.display_data()

    logger.info('### TRANSLATE DATA FOR VIEW')
    translater_obj.translate_data_for_view()
    translater_obj.display_data()

    logger.info('### WRITE DATA FOR VIEW')
    translater_obj.write_data_for_view()

    logger.info('### DATA FOR MODEL')
    translater_obj.create_data_for_model()

    logger.info('### TRANSLATE DATA FOR MODEL')
    translater_obj.translate_data_for_model()

    logger.info('### GET DATA FOR MODEL')
    data_for_model = translater_obj.get_data_for_model()
    post_processers = translater_obj.get_post_processers()

    # train
    trainer_obj = Trainer(**data_for_model, kernel=True)
    trainer_obj.read_config_text(config_text)

    logger.info('### FIT')
    trainer_obj.calc_estimator()

    logger.info('### WRITE ESTIMATOR DATA')
    trainer_obj.write_estimator_data()

    logger.info('### GET ESTIMATOR DATA')
    estimator_data = trainer_obj.get_estimator_data()

    # output
    outputer_obj = Outputer(
        **data_for_model, **estimator_data, **post_processers, kernel=True)
    outputer_obj.read_config_text(config_text)

    logger.info('### PREDICT')
    outputer_obj.predict_y()
    outputer_obj.calc_predict_df()

    logger.info('### WRITE PREDICT DATA')
    outputer_obj.write_predict_data()

    logger.info('# FINISHED')
