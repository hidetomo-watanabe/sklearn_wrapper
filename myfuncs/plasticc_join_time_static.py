import sys
import pandas as pd

if __name__ == '__main__':
    input_filename = sys.argv[1]
    input_meta_filename = sys.argv[2]
    output_filename = sys.argv[3]

    input_df = pd.read_csv(input_filename)
    grouped_df = input_df.groupby('object_id')
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
    input_static_df = mean_df
    input_static_df = input_static_df.join(std_df)
    input_static_df = input_static_df.join(min_df)
    input_static_df = input_static_df.join(max_df)

    # output
    output_df = pd.read_csv(input_meta_filename).set_index('object_id')
    output_df = output_df.join(input_static_df)
    output_df.to_csv(output_filename)
