import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import numpy as np
import warnings
import argparse
warnings.filterwarnings('ignore')
# sampling_rate = 0.2
# basedir = "dataset/csv/flow/"
# filename = "01-12-flow.csv"

basedir = "dataset/csv/bpf-flow/"
filename = "01-12-flow.csv"
df = pd.read_csv(f"{basedir}/{filename}")
df['Label'].value_counts()
attack_map = {
    'TFTP': 'Reflection',
    'SNMP': 'Reflection',
    'DNS': 'Reflection',
    'MSSQL': 'Reflection',
    'NetBIOS': 'Reflection',
    'UDP': 'Exploitation',
    'SSDP': 'Reflection',
    'LDAP': 'Reflection',
    'SYN': 'Exploitation', 
    'NTP': 'Reflection',
    'UDP-Lag': 'Exploitation',
    'Benign': 'Benign',
    'PortMap': 'Exploitation',
    'WebDDoS': 'Reflection'
}
df['label'] = df['Label'].map(attack_map)
attack_number_map = {
    'Benign': 0,
    'Reflection': 1,
    'Exploitation': 1
}
df['Attack_Number'] = df['label'].map(attack_number_map).astype(int)
df['Protocol'] = df['Protocol'].map({6: 1, 17: 0})
cols = ['Fwd Packet Length Max',
     'Fwd Packet Length Mean',
     'Fwd Packet Length Min',
     'Init_Win_bytes_forward',
     'ACK Flag Count',
     'Total Length of Fwd Packets',
     'FIN Flag Count',
     'Fwd IAT Mean',
     'Fwd IAT Total',
     'Fwd Packet Length Var',
     'SYN Flag Count',
     'Fwd IAT Max',
     'Attack_Number']
df = df[cols]
sampled_data_benign = df[df['Attack_Number'] == 0]
sample_size = int(0.2 * len(df)) - len(sampled_data_benign)
sampled_data = df.sample(n = sample_size, replace = False, random_state = 0)
numeric_cols = df.select_dtypes(include = np.number).columns
li_df = [sampled_data[numeric_cols], sampled_data_benign[numeric_cols]]
sampled_data = pd.concat(li_df)
sampled_data.to_csv(f"{basedir}/01-12-flow-reshaped-mlp.csv", index=False)

basedir = "dataset/csv/flow/"
filename = "01-12-flow.csv"
df = pd.read_csv(f"{basedir}/{filename}")
df['Label'].value_counts()
attack_map = {
    'TFTP': 'Reflection',
    'SNMP': 'Reflection',
    'DNS': 'Reflection',
    'MSSQL': 'Reflection',
    'NetBIOS': 'Reflection',
    'UDP': 'Exploitation',
    'SSDP': 'Reflection',
    'LDAP': 'Reflection',
    'SYN': 'Exploitation', 
    'NTP': 'Reflection',
    'UDP-Lag': 'Exploitation',
    'Benign': 'Benign',
    'PortMap': 'Exploitation',
    'WebDDoS': 'Reflection'
}
df['label'] = df['Label'].map(attack_map)
attack_number_map = {
    'Benign': 0,
    'Reflection': 1,
    'Exploitation': 1
}
df['Attack_Number'] = df['label'].map(attack_number_map).astype(int)
df['Protocol'] = df['Protocol'].map({6: 1, 17: 0})
cols = ['Fwd Packet Length Max',
     'Fwd Packet Length Mean',
     'Fwd Packet Length Min',
     'Init_Win_bytes_forward',
     'ACK Flag Count',
     'Total Length of Fwd Packets',
     'FIN Flag Count',
     'Fwd IAT Mean',
     'Fwd IAT Total',
     'Fwd Packet Length Var',
     'SYN Flag Count',
     'Fwd IAT Max',
     'Attack_Number']
df = df[cols]
sampled_data_benign = df[df['Attack_Number'] == 0]
sample_size = int(0.2 * len(df)) - len(sampled_data_benign)
sampled_data = df.sample(n = sample_size, replace = False, random_state = 0)
numeric_cols = df.select_dtypes(include = np.number).columns
li_df = [sampled_data[numeric_cols], sampled_data_benign[numeric_cols]]
sampled_data = pd.concat(li_df)
sampled_data.to_csv(f"{basedir}/01-12-flow-reshaped-mlp.csv", index=False)
