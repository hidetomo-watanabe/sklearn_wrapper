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
    print('[INFO] GET OBJECT IDS')
    object_ids = pd.read_csv(input_meta_filename)['object_id'].values
    # csvが大きすぎるため分割して処理
    print('[INFO] READ INPUT AND GROUPBY DIVIDEDLY')
    CHUNKSIZE = 5000000
    start_index = 0
    stat_df = pd.DataFrame()
    before_last_id = None
    process_bar = tqdm(total=len(object_ids))
    while True:
        # チェック開始indexまでskip
        # メモリ削減のため、読み込むrowを制限
        # 2 x CHUNKSIZEを読んで、またいだデータに対応
        # skiprowsが徐々に重くなるかも。。。
        if start_index == 0:
            input_reader = list(pd.read_csv(
                input_filename, chunksize=CHUNKSIZE,
                skiprows=start_index * CHUNKSIZE,
                nrows=2 * CHUNKSIZE))
        else:
            input_reader = list(pd.read_csv(
                input_filename, chunksize=CHUNKSIZE, header=1,
                names=[
                    'object_id',
                    'mjd',
                    'passband',
                    'flux',
                    'flux_err',
                    'detected'
                ],
                skiprows=start_index * CHUNKSIZE,
                nrows=2 * CHUNKSIZE))

        # ファイルを全て読んだため、loop終了
        if len(input_reader) == 0:
            break

        r0 = input_reader[0]
        # r0のうち、すでに処理したデータを除いたもの
        if before_last_id:
            input_df_part = r0[r0['object_id'] != before_last_id]
        else:
            input_df_part = r0
        if len(input_reader) > 1:
            r1 = input_reader[1]
            # r1の先頭のまたいだデータを追加
            r0_last_id = r0['object_id'].values[-1]
            input_df_part = pd.concat(
                [input_df_part, r1.loc[r1['object_id'] == r0_last_id]],
                ignore_index=True)
            del r0
        del r1

        # 処理対象がないため、loop終了
        if len(input_df_part) == 0:
            break

        # 次loop時に対象外にするID
        before_last_id = input_df_part['object_id'].values[-1]

        # groupby
        grouped_df_part = input_df_part.groupby('object_id')
        stat_df_part = _get_stat_df(grouped_df_part)
        stat_df = pd.concat([stat_df, stat_df_part], ignore_index=True)

        # 処理したobject_idの数だけprocess_barを更新
        process_bar.update(len(stat_df_part))
        start_index += 1

        # メモリ対策
        del input_df_part
        del grouped_df_part
        del stat_df_part
    process_bar.close()

    print('[INFO] JOIN OBJECT IDS')
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
