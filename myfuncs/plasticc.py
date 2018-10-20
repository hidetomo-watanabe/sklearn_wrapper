def translate_mjd(dfs, train_df):
    #######################################
    # mjd_diff = mjd_max - mjd_min
    #######################################
    for df in dfs:
        df['mjd_diff'] = df['mjd_max'] - df['mjd_min']
        del df['mjd_max']
        del df['mjd_min']
    return dfs


if __name__ == '__main__':
    pass
