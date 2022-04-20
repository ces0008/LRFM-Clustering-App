    # -- LRFM Clustering for Real Customers 

import pandas as pd
import numpy as np
import cx_Oracle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import yaml


def read_config(config_path=None):
    """ Read config file for used scrips and other configuration.
    :param config_path: Path of config.json. If None, read current dir.
    :return: data: Dictionary containing scripts for reading sql data.
    """
    if config_path is None:
        config_path = Path.cwd() / 'config.yaml'
    else:
        config_path = Path(config_path)

    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def read_table_from_db(config):
    script = config['table']
    print(script)
    conn_info = config['conn_info']
    if conn_info['service_name']:
        dsn = cx_Oracle.makedsn(host=conn_info['host'], port=conn_info['port'], service_name=conn_info['service_name'])
    elif conn_info['sid']:
        dsn = cx_Oracle.makedsn(host=conn_info['host'], port=conn_info['port'], sid=conn_info['sid'])
    else:
        raise ValueError('Options sid and service_name cannot be both empty')
    conn = cx_Oracle.connect(user=conn_info['username'], password=conn_info['password'], dsn=dsn)

    dataset = pd.read_sql(script, conn)
    return dataset


def convert_date(start, end):
    """
    Data convertor from integer "e.g. 140004" to number of passed month after min date of dataset.
    :param start: int, 6 digits date: smallest date
    :param end: int, 6 digit date: largest date
    :return: int, number of passed month from beginning. e.g. 1 for the smallest date
    """
    d_range = []
    d = start
    for i in range(60):
        d_range.append(d)
        if d == end:
            break
        if d % 100 == 12:
            d = int(str((d // 100) + 1) + '01')
        else:
            d = d + 1
    return {k: v for (k, v) in zip(d_range, range(1, len(d_range) + 3))}


def clustering(df, cities=None):
    if cities is None:
        cities = df.CUSTOMER_CITY.unique()
    df_list = []
    for city in cities:
        df_temp = df[df.CUSTOMER_CITY == city]
        if len(df_temp) < 100:
            del df_temp
            continue
        print(f'df.head: {df_temp.head(8)}')
        date_convert_list = convert_date(df_temp['MIN_TR_DATE'].min(), df_temp['MAX_TR_DATE'].max())
        df_temp['month_min'] = df_temp['MIN_TR_DATE'].replace(date_convert_list)
        df_temp['month_max'] = df_temp['MAX_TR_DATE'].replace(date_convert_list)
        print('max month: ' , df_temp['month_max'])
        today = df_temp['month_max'].max()
        df_temp['L'] = df_temp['month_max'] - df_temp['month_min']
        df_temp['R'] = today - df_temp['month_max']
        df_temp['R_CAT'] = pd.cut(df_temp['R'], bins=[-1, 3., 10., np.inf], labels=[1, 50, 70])
        df_temp['F'] = df_temp['SUM_C_CIRCUL_COUNT'] + df_temp['SUM_D_CIRCUL_COUNT']
        df_temp['M'] = df_temp['AVR_TOTAL']
        df_temp['R_CAT'] = df_temp.R_CAT.cat.codes
        features = df_temp[['L', 'R_CAT', 'F', 'M']].copy()
        features['F'] = features['F'].apply(lambda x: np.log(x + 1))
        features['M'] = features['M'].apply(lambda x: np.log(x + 1))
        scaler_std = MinMaxScaler()
        df_temp[['L_n', 'R_n', 'F_n', 'M_n']] = scaler_std.fit_transform(features)
        del scaler_std
        model_LR = KMeans(n_clusters=4)
        model_LR.fit_transform(df_temp[['L_n', 'R_n']])
        df_temp['label_lr'] = model_LR.predict(df_temp[['L_n', 'R_n']])
        del model_LR
        t_l = df_temp.groupby('label_lr')['L'].transform('mean') >= df_temp.L.mean()
        t_r = df_temp.groupby('label_lr')['R_CAT'].transform('mean') <= 1
        df_temp['Group'] = (t_l * 1).astype(str) + (t_r * 1).astype(str)
        group_map = {'11': 'مشتریان اصلی',
                     '10': 'مشتریان بالقوه',
                     '01': 'مشتریان جدید',
                     '00': 'مشتریان ازدست رفته'}
        df_temp['Group'].replace(group_map, inplace=True)
        CC_idx = df_temp[df_temp['Group'] == 'مشتریان اصلی'].index  # CC: Core Customer
        PC_idx = df_temp[df_temp['Group'] == 'مشتریان بالقوه'].index  # PC: Potential Customer
        NC_idx = df_temp[df_temp['Group'] == 'مشتریان جدید'].index  # NC: New Customer
        LC_idx = df_temp[df_temp['Group'] == 'مشتریان ازدست رفته'].index  # LC: Lost Customer
        index_list = [CC_idx, PC_idx, NC_idx, LC_idx]
        for i in index_list:
            if len(i) < 4:
                continue
            model_FM1 = KMeans(n_clusters=4)
            print(df_temp.loc[i].shape)
            print("i", i)
            model_FM1.fit_transform(df_temp.loc[i, ['F_n', 'M_n']])
            df_temp.loc[i, 'label_fm'] = model_FM1.predict(df_temp.loc[i, ['F_n', 'M_n']])
            del model_FM1
            t_f = df_temp.loc[i].groupby('label_fm')['F'].transform('mean') >= df_temp.loc[i].F.mean()
            t_m = df_temp.loc[i].groupby('label_fm')['M'].transform('mean') >= df_temp.loc[i].M.mean()
            df_temp.loc[i, 'Cluster'] = (t_f * 1).astype(str) + (t_m * 1).astype(str)

        CRC_L_idx = df_temp[((df_temp['Group'] == 'مشتریان اصلی') & (df_temp['Cluster'] == '00'))].index

        CRC_LR_idx = df_temp[(df_temp['Group'] == 'مشتریان بالقوه') & (df_temp['Cluster'] == '00')].index

        df_temp.loc[CRC_L_idx, 'Group'] = 'مشتریان معمولی'
        df_temp.loc[CRC_LR_idx, 'Group'] = 'مشتریان معمولی'
        CC_idx = df_temp[df_temp['Group'] == 'مشتریان اصلی'].index
        PC_idx = df_temp[df_temp['Group'] == 'مشتریان بالقوه'].index

        df_temp.loc[CC_idx, 'Cluster'] = df_temp.loc[CC_idx, 'Cluster'].replace({'11': 'مشتریان وفادار باارزش'
                                                                                    , '01': 'مشتریان طلایی',
                                                                                 '10': 'مشتریان وفادار پرتراکنش'})
        df_temp.loc[PC_idx, 'Cluster'] = df_temp.loc[PC_idx, 'Cluster'].replace({'11': 'مشتریان بالقوه وفادار',
                                                                                 '01': 'بالقوه با حجم تراکنش بالا',
                                                                                 '10': 'مشتریان بالقوه پرتراکنش'})
        df_temp.loc[NC_idx, 'Cluster'] = df_temp.loc[NC_idx, 'Cluster'].replace({'11': 'مشتریان جدید با ارزش',
                                                                                 '01': 'مشتریان جدید با حجم تراکنش بالا',
                                                                                 '10': 'مشتریان جدید پرتراکنش',
                                                                                 '00': 'مشتریان جدید نامطمئن'})
        df_temp.loc[LC_idx, 'Cluster'] = df_temp.loc[LC_idx, 'Cluster'].replace({'11': 'مشتریان ازدست رفته باارزش',
                                                                                 '01': 'مشتریان ازدست رفته حجم تراکنش بالا',
                                                                                 '10': 'مشتریان ازدست رفته پرتراکنش',
                                                                                 '00': 'مشتریان از دست رفته نامطمئن'})
        df_temp.loc[CRC_L_idx, 'Cluster'] = df_temp.loc[CRC_L_idx, 'Cluster'].replace({'00': 'مشتریان معمولی قدیمی'})

        df_temp.loc[CRC_LR_idx, 'Cluster'] = df_temp.loc[CRC_LR_idx, 'Cluster'].replace({'00': 'مشتریان معمولی همراه'})
        df_list.append(df_temp)
        del df_temp
    df_res = pd.concat(df_list)
    return df_res


def build_final_data(df, cities=None):
    df = clustering(df, cities)
    df = df[['ID', 'AGE', 'GROUP_NAME', 'CITY', 'SEX', 'NATIONALITY', 'MARRIAGE',
             'JOB', 'ECONOMICAL', 'POLITIC', 'EDUCATION', 'RELIGION',
             'AGE', 'CHILD', 'MIN_DATE', 'MAX_DATE', 'SUM_C_CIRCUL_COUNT',
             'SUM_D_CIRCUL_COUNT', 'AVR_TOTAL_AMOUNT', 'L', 'R', 'F', 'M', 'L_n', 'R_n', 'F_n', 'M_n',
             'Group', 'Cluster']]

    df_final = df.astype({'AGE': 'float64', 'L': 'float64', 'R': 'float64','F': 'float64', 'M': 'float64',
                          'SUM_C_CIRCUL_COUNT': 'float64','SUM_D_CIRCUL_COUNT': 'float64',
                          'AVR_TOTAL_AMOUNT': 'float64'
                          })
    df_final['lrfm_ex_date'] = df_final['MAX_TR_DATE'].max()
    return df_final


def run(config_path=None, cities=None):
    config = read_config(config_path)
    print(config)
    df = read_table_from_db(config)
    print(df.head())
    final_result = build_final_data(df, cities)
    print(final_result.head())
    output_path = Path(config['output_path'])
    final_result.to_csv(output_path, mode='a', index=False, encoding="utf-8-sig" ) # append to output_path 
    print(f'{output_path.name} is saved in {output_path.parent}')
    return print("Success!")
# -------------------------------------


if __name__ == '__main__':
    run()
