def translate_mjd(dfs, train_df):
    #######################################
    # mjd_diff = mjd_max - mjd_min
    #######################################
    for df in dfs:
        df['mjd_diff'] = df['mjd_max'] - df['mjd_min']
        del df['mjd_max']
        del df['mjd_min']
    return dfs


def translate_flux(dfs, train_df):
    #######################################
    # flux_diff = flux_max - flux_min
    #######################################
    for df in dfs:
        df['flux_diff'] = df['flux_max'] - df['flux_min']
        df['flux_diff2'] = df['flux_diff'] / df['flux_mean']
    return dfs


if __name__ == '__main__':
    pass
