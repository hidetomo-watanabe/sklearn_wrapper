import sys
from tqdm import tqdm
import pandas as pd

if __name__ == '__main__':
    input_filename = sys.argv[1]
    input_meta_filename = sys.argv[2]
    output_filename = sys.argv[3]

    def _get_stat_df(grouped_df):
        mean_df = grouped_df.mean().rename(
            columns={
                'mjd': 'mjd_mean',
                'passband': 'passband_mean',
                'flux': 'flux_mean',
                'flux_err': 'flux_err_mean',
                'detected': 'detected_mean',
            })
        std_df = grouped_df.std().rename(
            columns={
                'mjd': 'mjd_std',
                'passband': 'passband_std',
                'flux': 'flux_std',
                'flux_err': 'flux_err_std',
                'detected': 'detected_std',
            })
        min_df = grouped_df.min().rename(
            columns={
                'mjd': 'mjd_min',
                'passband': 'passband_min',
                'flux': 'flux_min',
                'flux_err': 'flux_err_min',
                'detected': 'detected_min',
            })
        max_df = grouped_df.max().rename(
            columns={
                'mjd': 'mjd_max',
                'passband': 'passband_max',
                'flux': 'flux_max',
                'flux_err': 'flux_err_max',
                'detected': 'detected_max',
            })
        stat_df = mean_df
        stat_df = stat_df.join(std_df)
        stat_df = stat_df.join(min_df)
        stat_df = stat_df.join(max_df)
        return stat_df

    print('[INFO] CREATE STAT')
    # csvが大きすぎるため分割して処理
    print('[INFO] GET OBJECT IDS')
    object_ids = pd.read_csv(input_meta_filename)['object_id'].values
    print('[INFO] READ INPUT AND GROUPBY ONE BY ONE')
    stat_df = pd.DataFrame()
    # object_idごとに、統計情報を計算していく
    # object_idの順番はinput_metaとinputで同一と仮定
    # object_idが進むごとにinput_readerのチェック開始indexを増やしていく
    input_reader = list(pd.read_csv(input_filename, chunksize=10000))
    read_start_index = 0
    for object_id in tqdm(object_ids):
        input_df_part = pd.DataFrame()
        # チェック開始indexまでskip
        for i, x in enumerate(input_reader[read_start_index:]):
            x_part = x.loc[x['object_id'] == object_id]
            if len(x_part) > 0:
                input_df_part = pd.concat(
                    [input_df_part, x_part], ignore_index=True)
            else:
                # object_idは固まって存在していると仮定
                # 該当データ群が途切れたらbreak
                if len(input_df_part) > 0:
                    read_start_index += i - 1
                    break
        grouped_df_part = input_df_part.groupby('object_id')
        stat_df_part = _get_stat_df(grouped_df_part)
        stat_df = pd.concat([stat_df, stat_df_part], ignore_index=True)
    # object_idの順番はinput_metaとinputで同一と仮定
    stat_df['object_id'] = object_ids
    stat_df = stat_df.set_index('object_id')

    print('[INFO] CREATE OUTPUT')
    print('[INFO] JOIN META')
    output_df = pd.read_csv(input_meta_filename).set_index('object_id')
    print('[INFO] JOIN STAT')
    output_df = output_df.join(stat_df)
    print('[INFO] WRITE OUTPUT TO CSV')
    output_df.to_csv(output_filename)
