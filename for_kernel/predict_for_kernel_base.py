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

    logger.info('### DATA FOR VIEW')
    translater_obj.create_data_for_view()
    translater_obj.display_data()

    logger.info('### TRANSLATE DATA FOR VIEW')
    translater_obj.translate_data_for_view()
    translater_obj.display_data()

    logger.info('##### DATA FOR MODEL')
    translater_obj.create_data_for_model()
    translater_obj.normalize_data_for_model()
    translater_obj.reduce_dimension_of_data_for_model()
    data_for_model = translater_obj.get_data_for_model()

    # predict
    predicter_obj = Predicter(**data_for_model, kernel=True)
    predicter_obj.read_config_text(config_text)

    logger.info('### VALIDATE')
    predicter_obj.is_ok_with_adversarial_validation()

    logger.info('### FIT')
    predicter_obj.calc_ensemble_model()

    logger.info('### PREDICT')
    predicter_obj.predict_y()
    predicter_obj.calc_predict_df()

    logger.info('### WRITE PREDICT DATA')
    predicter_obj.write_predict_data()

    logger.info('# FINISHED')
