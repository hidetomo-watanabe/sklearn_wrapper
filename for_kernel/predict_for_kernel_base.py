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

    logger.info('### RAW DATA')
    translater_obj.create_raw_data()
    translater_obj.display_data()

    logger.info('### TRANSLATE RAW DATA')
    translater_obj.translate_raw_data()
    translater_obj.display_data()

    logger.info('##### DATA FOR MODEL')
    translater_obj.create_data_for_model()
    translater_obj.normalize_data_for_model()
    translater_obj.reduce_dimension_of_data_for_model()
    feature_columns, test_ids, X_train, Y_train, X_test = \
        translater_obj.get_data_for_model()
    scaler_y = translater_obj.get_scaler_y()

    # predict
    predicter_obj = Predicter(
        feature_columns, test_ids,
        X_train, Y_train, X_test, scaler_y, kernel=True)
    predicter_obj.read_config_text(config_text)

    # logger.info('### VISUALIZE TRAIN DATA')
    # predicter_obj.visualize_train_data()

    logger.info('### VALIDATE')
    predicter_obj.is_ok_with_adversarial_validation()

    logger.info('### FIT')
    predicter_obj.calc_ensemble_model()

    # logger.info('### VISUALIZE LEARNING CURVES')
    # predicter_obj.visualize_learning_curves()

    logger.info('### PREDICT')
    predicter_obj.calc_output()

    # logger.info('### VISUALIZE TRAIN PRED DATA')
    # predicter_obj.visualize_train_pred_data()

    logger.info('### OUTPUT')
    predicter_obj.write_output()

    logger.info('# FINISHED')
