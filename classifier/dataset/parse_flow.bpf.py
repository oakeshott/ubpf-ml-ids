#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import warnings
import os
import re
import glob
warnings.filterwarnings("ignore")

def welford_var(values) -> float:
    """
    Welford's online algorithm による分散計算（母分散, ddof=0）。
    values: 1次元配列・リストなど
    """
    n = 0
    mean = 0.0
    M2 = 0.0

    for x in values:
        x = float(x)
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2

    if n == 0:
        return 0.0
    return M2 / n

def get_index(path: str) -> int:
    """ファイル名から先頭の数字を取り出して整数として返す"""
    base = os.path.basename(path)
    m = re.match(r"(\d+)_packets\.csv", base)
    if m:
        return int(m.group(1))
    # マッチしない場合は末尾に回す
    return 10**9
    # あるいは raise でもよい:
    # raise ValueError(f"Unexpected filename: {base}")

def proto_to_num(proto: str) -> int:
    """Protocol文字列を数値にマップ"""
    if isinstance(proto, str):
        p = proto.upper()
        if p == "TCP":
            return 6
        if p == "UDP":
            return 17
    return 0


def compute_flow_features(group: pd.DataFrame, obs_time: Optional[float]):
    """
    1フロー(5タプル)のパケット集合から特徴量を計算する.

    group: ある 5タプルで groupby された DataFrame
    obs_time: フロー観測時間(秒). None の場合は全パケット使用
    """
    # Timestamp でソート
    group = group.sort_values("Timestamp")
    if group.empty:
        return None

    # 観測時間でフィルタリング
    t0 = group["Timestamp"].iloc[0]
    if obs_time is not None:
        cutoff = t0 + pd.to_timedelta(obs_time, unit="s")
        group = group[group["Timestamp"] <= cutoff]
        if group.empty:
            return None

    n_packets = len(group)

    # 5タプル情報
    src_ip = group["Source IP"].iloc[0]
    dst_ip = group["Destination IP"].iloc[0]
    src_port = int(group["Source Port"].iloc[0])
    dst_port = int(group["Destination Port"].iloc[0])
    proto_num = int(group["ProtoNum"].iloc[0])

    # --- Length 系 ---
    lengths = group["Length"].astype(float).to_numpy()
    fwd_pkt_len_max  = float(np.max(lengths))
    fwd_pkt_len_min  = float(np.min(lengths))
    fwd_pkt_len_mean = float(np.mean(lengths))
    fwd_pkt_len_var  = float(welford_var(lengths)) if n_packets > 1 else 0.0
    total_len_fwd    = float(np.sum(lengths))

    # --- フロー継続時間 (ms) ---
    t_first = group["Timestamp"].iloc[0]
    t_last = group["Timestamp"].iloc[-1]
    flow_duration_ms = (t_last - t_first).total_seconds() * 1000.0

    # --- IAT (ms) ---
    times_ns = group["Timestamp"].view("int64")  # datetime64[ns] -> int64(ns)
    if n_packets > 1:
        iats_ms = np.diff(times_ns).astype(float) / 1e3  # ns -> ms
        fwd_iat_total = float(np.sum(iats_ms))
        fwd_iat_mean = float(np.mean(iats_ms))
        fwd_iat_max = float(np.max(iats_ms))
        fwd_iat_min = float(np.min(iats_ms))
        fwd_iat_var = float(welford_var(iats_ms))
    else:
        fwd_iat_total = 0.0
        fwd_iat_mean = 0.0
        fwd_iat_max = 0.0
        fwd_iat_min = 0.0
        fwd_iat_var = 0.0

    # --- TCP Window と Flags カウント ---
    tcp_rows = group[group["Protocol"].str.upper() == "TCP"]
    if not tcp_rows.empty:
        # Init_Win_bytes_forward: 最初の TCP パケットの Window size
        init_win_fwd = int(tcp_rows["Window size"].iloc[0])

        flags_series = tcp_rows["Flags"].astype(str)
        ack_cnt = int((flags_series.str.contains("A")).sum())
        syn_cnt = int((flags_series.str.contains("S")).sum())
        rst_cnt = int((flags_series.str.contains("R")).sum())
        fin_cnt = int((flags_series.str.contains("F")).sum())
    else:
        init_win_fwd = 0
        ack_cnt = syn_cnt = rst_cnt = fin_cnt = 0

    # --- Fwd Header Length (暫定実装) ---
    # 本来は IP/TCPヘッダ長から計算すべきだが、CSVに情報がないため
    # 1パケットあたり32バイトのヘッダ長と仮定
    fwd_header_len = int(n_packets * 32)

    # --- フローラベル (多数決) ---
    labels = group["Label"].astype(str).to_numpy()
    uniques, counts = np.unique(labels, return_counts=True)
    main_label = uniques[np.argmax(counts)]

    return {
        "Source IP": src_ip,
        "Source Port": src_port,
        "Destination IP": dst_ip,
        "Destination Port": dst_port,
        "Protocol": proto_num,  # 数値プロトコル

        # "Flow Duration": flow_duration_ms,
        "Total Fwd Packets": int(n_packets),
        "Total Length of Fwd Packets": int(total_len_fwd),

        "Fwd Packet Length Max":  int(fwd_pkt_len_max),
        "Fwd Packet Length Min":  int(fwd_pkt_len_min),
        "Fwd Packet Length Mean": int(fwd_pkt_len_mean),
        "Fwd Packet Length Var":  int(fwd_pkt_len_var),

        "Fwd IAT Total": int(fwd_iat_total),
        "Fwd IAT Mean":  int(fwd_iat_mean),
        "Fwd IAT Max":   int(fwd_iat_max),
        "Fwd IAT Min":   int(fwd_iat_min),
        "Fwd IAT Var":   int(fwd_iat_var),

        "Fwd Header Length": int(fwd_header_len),
        "Init_Win_bytes_forward": int(init_win_fwd),

        "ACK Flag Count": int(ack_cnt),
        "SYN Flag Count": int(syn_cnt),
        "RST Flag Count": int(rst_cnt),
        "FIN Flag Count": int(fin_cnt),

        "Label": main_label,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ディレクトリ内のパケットCSVから5タプルフロー特徴量+ラベルを生成"
    )
    parser.add_argument(
        "input_dir",
        help="入力パケットCSVが保存されているディレクトリ",
    )
    parser.add_argument(
        "output_csv",
        help="出力フロー特徴量CSVファイル",
    )
    parser.add_argument(
        "--obs-time",
        type=float,
        default=None,
        help="フロー観測時間(秒)。指定しない場合は各フローの全パケットを使用",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"ERROR: {input_dir} はディレクトリではありません")

    # ディレクトリ内の *.csv をすべて対象にする
    csv_files = sorted(
            [p for p in input_dir.glob("*.csv")],
            key=get_index
            )
    if not csv_files:
        raise SystemExit(f"ERROR: {input_dir} に CSV ファイルが見つかりません")

    dfs = []
    for path in csv_files:
        print(f"Loading {path} ...")
        df_i = pd.read_csv(path)
        df_i["source_file"] = path.name  # どのファイル由来かを残したい場合に便利
        dfs.append(df_i)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total packets loaded: {len(df)}")

    # Timestamp を datetime に変換
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Protocol を数値に変換して ProtoNum に保存
    df["ProtoNum"] = df["Protocol"].apply(proto_to_num)

    # groupby用の5タプル（Protocolは文字列）
    group_cols = [
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
        # ファイルごとにフローを分けたい場合は "source_file" もここに追加:
        # "source_file",
    ]

    features = []
    for key, group in tqdm(df.groupby(group_cols, sort=False)):
        feat = compute_flow_features(group, args.obs_time)
        if feat is not None:
            features.append(feat)

    out_df = pd.DataFrame(features)
    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out_df)} flows to {args.output_csv}")


if __name__ == "__main__":
    main()
