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

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--sampled_indices', type=str, default=None, help='Filename of sampled indices')
# arg_parser.add_argument('--sampling_rate', type=float, default=0.2, help='Sampling rate for reshaping the dataset')
arg_parser.add_argument('--in-file', type=str, default="dataset/csv/flow/03-11-flow.csv", help='Filename of the dataset')
arg_parser.add_argument('--out-file', type=str, default="dataset/csv/flow/shaped_03-11-flow.csv", help='Filename of the dataset')
arg_parser.add_argument('--out-sampled_indices', type=str, default="sampled_indices.csv", help='Filename of the dataset')
args = arg_parser.parse_args()

def main():
    df = pd.read_csv(f"{args.in_file}")
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
    sample_size = len(sampled_data_benign)
    if args.sampled_indices == None:
        sampled_data = df[df['Attack_Number'] == 1].sample(n = sample_size, replace = False, random_state = 0)
        sampled_data.reset_index()[["index"]].to_csv(f"{args.out_sampled_indices}")
    else:
        sampled_indices = pd.read_csv(args.sampled_indices)
        sampled_data = df.iloc[sampled_indices["index"].values]
        print(sampled_indices["index"].values)
    df = pd.concat([sampled_data, sampled_data_benign])
    df = df.sample(frac=1, random_state=42)
    df.to_csv(f"{args.out_file}", index=False)
if __name__ == "__main__":
    main()
