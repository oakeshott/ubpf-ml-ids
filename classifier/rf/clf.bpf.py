#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import ctypes as ct
from ctypes import CDLL, POINTER
from ctypes import c_size_t, c_int32
import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def ensure_contiguous(array):
    """
    Ensure that array is contiguous
    :param array:
    :return:
    """
    return np.ascontiguousarray(array) if not array.flags['C_CONTIGUOUS'] else array

def load_train_test(train_csv: str, test_csv: str, target_col: str):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if target_col not in train_df.columns:
        raise ValueError(f"train_csv に {target_col} が存在しません。")
    if target_col not in test_df.columns:
        raise ValueError(f"test_csv に {target_col} が存在しません。")

    X_train_full = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    X_train_full = (X_train_full * 2**16).round()
    X_test = (X_test * 2**16).round()

    y_train_series = train_df[target_col]
    y_test_series = test_df[target_col]

    y_train_full = y_train_series.values.astype(np.int64)
    y_test = y_test_series.values.astype(np.int64)

    return X_train_full, y_train_full, X_test, y_test


def main():
    parser = argparse.ArgumentParser(
        description="DecisionTreeClassifier (train/test from different CSVs)"
    )
    parser.add_argument("--train-csv", type=str, required=True, help="学習用CSV（train+val）")
    parser.add_argument("--test-csv", type=str, required=True, help="テスト用CSV")
    parser.add_argument("--target", type=str, required=True, help="目的変数（ラベル）の列名")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="学習CSVのうちValidationに回す割合")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="木の最大深さ (default: None)")
    parser.add_argument("--criterion", type=str, default="gini",
                        choices=["gini", "entropy", "log_loss"],
                        help="分割規準 (default: gini)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="乱数シード (default: 42)")
    parser.add_argument("--resdir", type=str, default="results",
                        help="Output directory")
    args = parser.parse_args()

    print(f"[INFO] Loading train from {args.train_csv}")
    print(f"[INFO] Loading test  from {args.test_csv}")
    X_train_full, y_train_full, X_test, y_test = load_train_test(
        args.train_csv, args.test_csv, args.target
    )
    print(f"[INFO] X_train_full shape = {X_train_full.shape}, y_train_full shape = {y_train_full.shape}")
    print(f"[INFO] X_test shape       = {X_test.shape}, y_test shape       = {y_test.shape}")

    # train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_ratio,
        random_state=args.random_state,
        stratify=y_train_full
    )
    print(f"[INFO] Train size = {X_train.shape[0]}, Val size = {X_val.shape[0]}, Test size = {X_test.shape[0]}")

    lib = CDLL("./rf.so")
    rf = lib.rf

    c_uint_p = ct.POINTER(ct.c_uint)
    rf.argstypes = [POINTER(ct.c_longlong), ct.c_uint]
    rf.restype = None

    # === Test ===
    X_test = X_test.values.astype(np.int64)
    y_test = y_test
    N = 1
    y_test_pred = []
    for sample in tqdm(X_test):
        # print(sample)
        sample = np.ascontiguousarray(sample, dtype=np.int64)
        class_indices = np.zeros(N, dtype=np.uintc)
        class_indices = np.ascontiguousarray(class_indices)
        shape = class_indices.shape
        class_indices = class_indices.ctypes.data_as(c_uint_p)
        sample = sample.ctypes.data_as(POINTER(ct.c_longlong))
        rf(sample, class_indices)
        pred = np.ctypeslib.as_array(class_indices, shape)
        y_test_pred.append(pred[0])
    y_test_pred = np.array(y_test_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"[RESULT] Test Accuracy: {test_acc:.4f}\n")
    print("[RESULT] Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    ret = classification_report(y_test, y_test_pred, digits=6, output_dict=True)
    os.makedirs(args.resdir, exist_ok=True)
    with open(f"{args.resdir}/classification_report.bpf.json", "w") as f:
        json.dump(ret, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

if __name__ == "__main__":
    main()
