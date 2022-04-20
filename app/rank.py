import pandas as pd
import numpy as np
import cx_Oracle
from pathlib import Path
import yaml
import warnings
from datetime import datetime as dt
warnings.filterwarnings('ignore')


def read_config(config_path=None):
    """ Read config file for used scrips and other configuration.
    :param config_path: Path of config.json. If None, read current dir.
    :return: data: Dictionary containing scripts for reading sql data.
    """
    if config_path is None:
        config_path = Path.cwd() / 'config_rank.yaml'
    else:
        config_path = Path(config_path)

    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def read_table_from_db(config):
    script_clv = config['clv_table']
    conn_info = config['conn_info']
    if conn_info['service_name']:
        dsn = cx_Oracle.makedsn(host=conn_info['host'], port=conn_info['port'], service_name=conn_info['service_name'])
    elif conn_info['sid']:
        dsn = cx_Oracle.makedsn(host=conn_info['host'], port=conn_info['port'], sid=conn_info['sid'])
    else:
        raise ValueError('Options sid and service_name cannot be both empty')
    conn = cx_Oracle.connect(user=conn_info['username'], password=conn_info['password'], dsn=dsn)
    table_income = pd.read_sql(script_clv, con=conn)
    return table_income


def clv_ranking(config):
    clustering_path = config['input_path']
    df_final = pd.read_csv(clustering_path)
    #df_final = df_final.drop(labels='Unnamed: 0', axis=1)
    income_agg = read_table_from_db(config).groupby('CUSTOMER_ID').agg(INCOME=('INCOME', 'sum'))
    # maximum execution date of lrfm:
    last_exec_date = pd.to_numeric(df_final['lrfm_ex_date'], errors='coerce').astype('Int64').max()
    # select only latest rows
    df_final = df_final.loc[df_final['lrfm_ex_date'] == last_exec_date]
    df_final = df_final.merge(income_agg, left_on='ID', right_on='CUSTOMER_ID', how='inner')
    df_final = df_final[['ID', 'Group', 'Cluster', 'INCOME', 'CUSTOMER_CITY']]
    df_final['INCOME'] = df_final['INCOME'] / 10_000_000_000
    df_final.rename(columns={'INCOME': 'CLV in Milliard Toman'}, inplace=True)
    df_final['Percent of Total Group'] = df_final.groupby('Group')['ID'].transform('count') / len(df_final)
    df_final['Percent of Total Group'] = df_final['Percent of Total Group'].round(2) * 100
    df_final['Percent of Total Cluster'] = df_final.groupby('Cluster')['ID'].transform('count') / len(df_final)
    df_final['Percent of Total Cluster'] = df_final['Percent of Total Cluster'].round(2) * 100
    df_final['exec_date'] = last_exec_date
    return df_final


def run(config_path=None):
    config = read_config(config_path)
    print("printing config file info: \n\n\n", config, "\n")
    output_path = Path(config['output_path'])
    final_result = clv_ranking(config)
    final_result.to_csv(output_path, mode='a', index=False, encoding="utf-8-sig" ) # append to output_path 
    print(f'{output_path.name} is saved on {output_path.parent}', "\n\n")
    return print("Success!")
# -------------------------------------


if __name__ == '__main__':
    run()
